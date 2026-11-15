"""The workflow app builds its components from the workflow file, seen through its HTTP API.

A component takes any name and a stage calls it by that name: two rerankers of
the workflow file, each on its own endpoint, are called by two rerank stages,
each its own. The endpoints are an HTTP transport that answers in their place.
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

from rag_components import FAKE_BACKENDS
from rag_graphs import packaged_graph
from rag_load_test.composition.composition_root import CompositionRoot
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.config.workflow_file import PACKAGED_WORKFLOW
from rag_load_test.entrypoints.rag_serve import create_app
from repo_paths import REPO_ROOT

SettingsFactory = Callable[..., RagSettings]
QUESTION = "How many paid vacation days do employees accrue each year?"
COARSE_URL, FINE_URL = "http://coarse.test/v3/rerank", "http://fine.test/v3/rerank"


def two_rerank_stages() -> dict[str, Any]:
    """The packaged graph with a coarse rerank stage before its rerank stage."""
    graph = packaged_graph()
    graph["stages"]["rerank_coarse"] = {"type": "rerank", "call": "coarse", "top_k": 15}
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
        (FINE_URL, "crossencoder-camembert-base-mmarcoFR", 20),
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


def test_the_packaged_file_builds_the_rag_from_its_three_model_components(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The file's embedder, reranker and LLM, each moved to its fake by its path.
    monkeypatch.setenv("RAG_COMPONENTS__EMBEDDER__BACKEND", "fake")
    monkeypatch.setenv("RAG_COMPONENTS__RERANKER__BACKEND", "fake")
    monkeypatch.setenv("RAG_COMPONENTS__LLM__BACKEND", "fake")
    settings = settings_factory(vector_store_backend="fake")

    with TestClient(create_app(CompositionRoot(settings))) as app:
        response = app.post("/query", json={"question": QUESTION})
        ready = app.get("/readyz")

    assert response.status_code == 200, response.text
    assert response.json()["answer"]
    checks = [check["name"] for check in ready.json()["checks"]]
    assert sorted(checks) == ["embedder", "reranker", "vector_store"]


def api_keys_in(node: object) -> list[str]:
    """The ``api_key`` keys a YAML document holds, at any depth."""
    if isinstance(node, dict):
        own = [str(key) for key in node if key == "api_key"]
        return own + [k for value in node.values() for k in api_keys_in(value)]
    if isinstance(node, list):
        return [k for value in node for k in api_keys_in(value)]
    return []


@pytest.mark.parametrize(
    "path", [PACKAGED_WORKFLOW, REPO_ROOT / "README.md"], ids=["packaged", "readme"]
)
def test_no_example_writes_an_api_key_in_a_workflow_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    blocks = text.split("```yaml")[1:] if path.suffix == ".md" else [text]
    documents = [yaml.safe_load(block.split("```")[0]) for block in blocks]

    assert [key for doc in documents for key in api_keys_in(doc)] == []
