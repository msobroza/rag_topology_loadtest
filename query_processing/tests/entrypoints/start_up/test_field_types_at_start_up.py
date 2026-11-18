"""The types and labels of the workflow state fields, checked when ``qp-serve`` starts.

Each refusal case writes the fixture's graph with one mistake that start-up
once accepted and every request then failed on: a labelled field with a second
writer, a response field written as another type, a placeholder on a field that
holds no messages, an edge condition that reads a field of another type. Each
is a usage error before anything opens, naming what was found and what was
expected. A request attribute that accepts ``None`` reads a field of its other
type, so a plain call on a vector store starts and answers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from query_processing.config.files.prompts_file import PROMPTS_FILE_VAR
from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.entrypoints import qp_serve
from rag_components import FAKE_BACKEND_VARIABLES, shell_settings
from rag_graphs import (
    FIXTURE_PROMPTS,
    fake_router,
    fixture_components,
    fixture_graph,
    write_routes_file,
    write_workflow_file,
)

QUESTION = "How many paid vacation days do employees accrue each year?"
Graph = dict[str, Any]


@pytest.fixture
def served_apps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Any]:
    """All-fake backends, no QP_* from the shell, an empty directory, no real server.

    Returns:
        The apps ``qp-serve`` asked uvicorn to serve.
    """
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    for name, value in FAKE_BACKEND_VARIABLES.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    apps: list[Any] = []
    monkeypatch.setattr(qp_serve.uvicorn, "run", lambda app, **kw: apps.append(app))
    return apps


def start_on(
    graph: Graph, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **components: Any
) -> int:
    """Run ``qp-serve`` on ``graph``, with the fixture's components and ``components``."""
    workflow = write_workflow_file(
        tmp_path / "workflow.yaml", graph, {**fixture_components(), **components}
    )
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))
    return qp_serve.main([])


def refusal_of(
    graph: Graph,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    **components: Any,
) -> str:
    """Run ``start_on``, which must stop with a usage error; return what it said."""
    with pytest.raises(SystemExit) as exited:
        start_on(graph, monkeypatch, tmp_path, **components)
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: qp-serve")
    assert "Traceback" not in err
    return " ".join(err.split())  # argparse wraps its message


def test_a_routed_field_that_a_stage_writes_again_without_labels_is_refused(
    served_apps: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The router writes route with its routes; a generate stage writes it again,
    # with any text, and the label edge would meet a label it has no row for.
    write_routes_file(tmp_path / "routes.yaml", ["a", "b"])
    graph = fixture_graph()
    graph["stages"]["choose_route"] = {
        "call": "router",
        "reads": ["query_embedding"],
        "writes": ["route"],
    }
    graph["stages"]["relabel"] = {
        "type": "generate",
        "call": "llm",
        "prompt": "answer_from_context",
        "reads": ["conversational_query", "reranked"],
        "writes": ["route"],
    }
    graph["edges"] = [
        ["START", "embed", "choose_route"],
        ["retrieve", "rerank", "relabel", "generate", "END"],
    ]
    graph["conditional_edges"] = [
        {
            "source": "choose_route",
            "path": "label",
            "reads": ["route"],
            "path_map": {"a": "retrieve", "b": "END"},
        }
    ]

    refusal = refusal_of(
        graph, monkeypatch, tmp_path, capsys, router=fake_router("embedder", ["a", "b"])
    )

    assert (
        "'route' is written with the labels 'a', 'b' by 'choose_route' and with "
        "no labels by 'relabel' (expected every writer of a field to declare the same labels)"
    ) in refusal
    assert served_apps == []


def test_a_response_field_written_as_another_type_is_refused(
    served_apps: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    graph = fixture_graph()
    del graph["stages"]["generate"]
    graph["stages"]["embed_answer"] = {
        "call": "embedder",
        "reads": ["standalone_query"],
        "writes": ["answer"],
    }
    graph["edges"][1] = ["embed_answer", "END"]
    graph["conditional_edges"][0]["path_map"][True] = "embed_answer"

    refusal = refusal_of(graph, monkeypatch, tmp_path, capsys)

    assert (
        "the response reads 'answer' as str | None, but 'embed_answer' writes it "
        "as list[float] (expected a field written as str | None)"
    ) in refusal
    assert served_apps == []


def test_a_placeholder_on_a_field_that_holds_no_messages_is_refused(
    served_apps: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompts = yaml.safe_load(FIXTURE_PROMPTS.read_text(encoding="utf-8"))
    prompts["answer_from_context"].append({"placeholder": "conversational_query"})
    prompts_file = tmp_path / "prompts.yaml"
    prompts_file.write_text(yaml.safe_dump(prompts), encoding="utf-8")
    monkeypatch.setenv(PROMPTS_FILE_VAR, str(prompts_file))

    refusal = refusal_of(fixture_graph(), monkeypatch, tmp_path, capsys)

    assert (
        "stage 'generate' reads 'conversational_query' as list[HistoryMessage], "
        "but the request writes it as str"
    ) in refusal
    assert served_apps == []


def test_an_edge_condition_that_reads_a_field_of_another_type_is_refused(
    served_apps: list[Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    graph = fixture_graph()
    graph["conditional_edges"][0]["reads"] = ["conversational_history"]

    refusal = refusal_of(graph, monkeypatch, tmp_path, capsys)

    assert (
        "conditional edge from 'rerank' on 'is_query_mode' reads "
        "'conversational_history' as str, but the request writes it as "
        "list[HistoryMessage] (expected a field written as str)"
    ) in refusal
    assert served_apps == []


def test_a_request_attribute_that_accepts_none_reads_a_field_of_its_other_type(
    served_apps: list[Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The store's request takes embedding: list[float] | None, text: str | None.
    graph = fixture_graph()
    graph["stages"]["search"] = {
        "call": "vector_store",
        "retrieval_mode": "hybrid",
        "top_k": 20,
        "rrf_k": 60,
        "reads": ["query_embedding", "standalone_query", "metadata_filter"],
        "writes": ["candidates"],
    }
    del graph["stages"]["retrieve"]
    graph["edges"][0] = ["START", "embed", "search", "rerank"]

    assert start_on(graph, monkeypatch, tmp_path) == 0

    [app] = served_apps
    with TestClient(app) as client:
        response = client.post("/query", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    assert response.json()["answer"]
