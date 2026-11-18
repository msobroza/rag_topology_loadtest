"""The workflow app builds its components from the workflow file, seen through its HTTP API.

A component takes any name and a stage calls it by that name: two rerankers of
the workflow file, each on its own endpoint, are called by two rerank stages,
each its own. The endpoints are an HTTP transport that answers in their place.
The vector store is a component of the file too, its fake backend's seed size
read from its block.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.retrieval.corpus import chunk, synthetic_corpus
from query_processing.entrypoints.qp_serve import create_app
from rag_components import FAKE_BACKENDS
from rag_graphs import (
    FIXTURE_WORKFLOW,
    fixture_components,
    fixture_graph,
    write_workflow_file,
)
from repo_paths import REPO_ROOT

SettingsFactory = Callable[..., WorkflowSettings]
QUESTION = "How many paid vacation days do employees accrue each year?"
COARSE_URL, FINE_URL = "http://coarse.test/v3/rerank", "http://fine.test/v3/rerank"


def two_rerank_stages() -> dict[str, Any]:
    """The fixture's graph with a coarse rerank stage before its rerank stage."""
    graph = fixture_graph()
    graph["stages"]["rerank_coarse"] = {**graph["stages"]["rerank"], "call": "coarse"}
    graph["stages"]["rerank_coarse"]["top_k"] = 15
    graph["edges"][0] = ["START", "embed", "retrieve", "rerank_coarse", "rerank"]
    return graph


class RerankEndpoints:
    """Both rerank endpoints: each answers every document, the last ones best."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, int]] = []  # URL, model, documents

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        documents = body["documents"]
        self.sent.append((str(request.url), body["model"], len(documents)))
        results = [
            {"index": i, "relevance_score": float(i)} for i in range(len(documents))
        ]
        return httpx.Response(200, json={"results": results})


def test_two_stages_call_two_components_of_one_type(
    settings_factory: SettingsFactory,
) -> None:
    coarse = {
        "type": "reranker",
        "backend": "api",
        "source_id": "org/coarse-rr",
        "api": {"url": COARSE_URL},
    }
    settings = settings_factory(
        **FAKE_BACKENDS | {"components__reranker__backend": "api"},
        components__reranker__api__url=FINE_URL,
        components={"coarse": coarse},
        graph=two_rerank_stages(),
    )
    endpoints = RerankEndpoints()
    composition = CompositionRoot(settings, transport=httpx.MockTransport(endpoints))

    with TestClient(create_app(composition)) as app:
        response = app.post("/retrieve", json={"question": QUESTION})

    assert response.status_code == 200, response.text
    # Each stage called its own component, under that component's alias.
    assert endpoints.sent == [
        (COARSE_URL, "coarse-rr", 20),
        (FINE_URL, "fixture-reranker", 20),
    ]
    timings = response.json()["timings_ms"]
    assert list(timings) == [
        "embed",
        "retrieve",
        "rerank",
        "generate",
        "rerank_coarse",
        "total",
    ]
    assert len(response.json()["passages"]) == 10  # the rerank stage's top_k


def test_the_fixture_file_builds_a_graph_from_its_three_model_components(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The file's embedder, reranker and LLM, each moved to its fake by its path.
    monkeypatch.setenv("QP_COMPONENTS__EMBEDDER__BACKEND", "fake")
    monkeypatch.setenv("QP_COMPONENTS__RERANKER__BACKEND", "fake")
    monkeypatch.setenv("QP_COMPONENTS__LLM__BACKEND", "fake")
    settings = settings_factory(components__vector_store__backend="fake")

    with TestClient(create_app(CompositionRoot(settings))) as app:
        response = app.post("/query", json={"question": QUESTION})
        ready = app.get("/readyz")

    assert response.status_code == 200, response.text
    assert response.json()["answer"]
    checks = [check["name"] for check in ready.json()["checks"]]
    assert sorted(checks) == ["embedder", "reranker", "vector_store"]


def passages_of(documents: int) -> int:
    """How many passages the fake store holds once seeded with ``documents``."""
    return sum(len(chunk(document)) for document in synthetic_corpus(documents))


def write_file_with_store(
    path: Path, monkeypatch: pytest.MonkeyPatch, name: str, store: dict[str, Any]
) -> None:
    """Write a workflow file of the fixture's graph, every model on its fake, and ``store``.

    The retrieve stage calls the store by ``name``; the file names it the file
    to read.
    """
    models = {n: {**c, "backend": "fake"} for n, c in fixture_components().items()}
    del models["vector_store"]
    graph = fixture_graph()
    graph["stages"]["retrieve"]["call"] = name
    write_workflow_file(path, graph, {**models, name: store})
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(path))


def test_the_vector_store_is_built_from_its_entry_in_the_workflow_file(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = {"backend": "fake", "fake": {"seed_documents": 3}}
    write_file_with_store(
        tmp_path / "workflow.yaml", monkeypatch, "vector_store", store
    )

    with TestClient(create_app(CompositionRoot(settings_factory()))) as app:
        response = app.post("/retrieve", json={"question": QUESTION})
        ready = app.get("/readyz").json()

    assert response.status_code == 200, response.text
    [check] = [c for c in ready["checks"] if c["name"] == "vector_store"]
    assert check["detail"] == f"{passages_of(3)} passages in memory"


def test_a_store_of_any_name_is_called_by_that_name_and_read_by_its_path(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = {"type": "vector_store", "backend": "fake", "fake": {"seed_documents": 3}}
    write_file_with_store(tmp_path / "workflow.yaml", monkeypatch, "passages", store)
    monkeypatch.setenv("QP_COMPONENTS__PASSAGES__FAKE__SEED_DOCUMENTS", "5")

    with TestClient(create_app(CompositionRoot(settings_factory()))) as app:
        response = app.post("/retrieve", json={"question": QUESTION})
        ready = app.get("/readyz").json()

    assert response.status_code == 200, response.text
    [check] = [c for c in ready["checks"] if c["name"] == "passages"]
    assert check["detail"] == f"{passages_of(5)} passages in memory"


def api_keys_in(node: object) -> list[str]:
    """The ``api_key`` keys a YAML document holds, at any depth."""
    if isinstance(node, dict):
        own = [str(key) for key in node if key == "api_key"]
        return own + [k for value in node.values() for k in api_keys_in(value)]
    if isinstance(node, list):
        return [k for value in node for k in api_keys_in(value)]
    return []


@pytest.mark.parametrize(
    "path", [FIXTURE_WORKFLOW, REPO_ROOT / "README.md"], ids=["fixture", "readme"]
)
def test_no_example_writes_an_api_key_in_a_workflow_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    blocks = text.split("```yaml")[1:] if path.suffix == ".md" else [text]
    documents = [yaml.safe_load(block.split("```")[0]) for block in blocks]

    assert [key for doc in documents for key in api_keys_in(doc)] == []
