"""A workflow graph that cannot run is refused when ``rag-serve`` starts.

Each case writes the packaged graph with one mistake into a workflow file and
runs the command. The refusal is a usage error naming what is wrong and what
was expected, raised before the server starts, so before any component opens;
nothing is left to fail on the first request. What only an opened component
shows, a store that ranks no text under a lexical graph, stops the start-up
with its message alone in the server's log, not a traceback.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from uvicorn.lifespan.on import LifespanOn

from rag_graphs import packaged_graph, rag_graph, write_workflow_file
from rag_load_test.composition.composition_root import CompositionRoot
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.config.workflow_file import WORKFLOW_FILE_VARIABLE
from rag_load_test.entrypoints import rag_serve
from rag_load_test.entrypoints.rag_serve import GraphRefusalInOneLine, create_app
from workflow_fakes import DenseOnlyStore

QUESTION = "How many paid vacation days do employees accrue each year?"
FAKE_BACKENDS = {
    "RAG_EMBEDDING_BACKEND": "fake",
    "RAG_RERANK_BACKEND": "fake",
    "RAG_GENERATION_BACKEND": "fake",
    "RAG_VECTOR_STORE_BACKEND": "fake",
}
STAGE_TYPES = "'embed', 'generate', 'rerank', 'retrieve'"
COMPONENTS = "'embedder', 'llm', 'reranker', 'vector_store'"
Graph = dict[str, Any]


class FakeServer:
    """Stands in for ``uvicorn.run``: records the app it was asked to serve."""

    def __init__(self) -> None:
        self.apps: list[Any] = []

    def run(self, app: Any, **options: object) -> None:
        self.apps.append(app)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeServer:
    """All-fake backends, no RAG_* from the shell, an empty directory, no real server."""
    for name in [n for n in os.environ if n.startswith(("RAG_", "DOMINO_"))]:
        monkeypatch.delenv(name)
    for name, value in FAKE_BACKENDS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    fake = FakeServer()
    monkeypatch.setattr(rag_serve.uvicorn, "run", fake.run)
    return fake


def start_on(graph: Graph, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> int:
    """Run ``rag-serve`` on a workflow file holding ``graph``; return its status."""
    workflow = write_workflow_file(tmp_path / "workflow.yaml", graph)
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))
    return rag_serve.main([])


# --- one mistake each: the change, then what the refusal says ----------------------


def unknown_stage_type(graph: Graph) -> None:
    graph["stages"]["rerank"]["type"] = "rerankk"


def missing_component(graph: Graph) -> None:
    graph["stages"]["rerank"]["call"] = "rerankr"


def component_of_the_wrong_type(graph: Graph) -> None:
    graph["stages"]["rerank"]["call"] = "llm"


def dense_retrieval_with_no_embed_stage(graph: Graph) -> None:
    del graph["stages"]["embed"]
    graph["edges"][0] = ["START", "retrieve", "rerank"]


def embed_stage_off_one_path(graph: Graph) -> None:
    # The question goes to embed only in query mode: a /retrieve request would
    # reach retrieve with no vector.
    graph["edges"][0] = ["embed", "retrieve", "rerank"]
    path_map = {True: "embed", False: "retrieve"}
    graph["conditional_edges"].append(
        {"source": "START", "path": "is_query_mode", "path_map": path_map}
    )


def edge_to_an_unknown_stage(graph: Graph) -> None:
    graph["edges"][1] = ["generate", "ending"]


def edge_from_an_unknown_stage(graph: Graph) -> None:
    graph["edges"][1] = ["generat", "END"]


def unregistered_edge_condition(graph: Graph) -> None:
    graph["conditional_edges"][0]["path"] = "is_query"


def conditional_edge_without_a_false_target(graph: Graph) -> None:
    del graph["conditional_edges"][0]["path_map"][False]


def conditional_edge_without_a_true_target(graph: Graph) -> None:
    del graph["conditional_edges"][0]["path_map"][True]


def conditional_edge_to_an_unknown_stage(graph: Graph) -> None:
    graph["conditional_edges"][0]["path_map"][True] = "generat"


def edge_condition_twice_on_one_source(graph: Graph) -> None:
    graph["conditional_edges"].append(dict(graph["conditional_edges"][0]))


def stage_named_like_the_total(graph: Graph) -> None:
    graph["stages"]["total"] = graph["stages"].pop("generate")
    graph["stages"]["total"]["type"] = "generate"
    graph["edges"][1] = ["total", "END"]
    graph["conditional_edges"][0]["path_map"][True] = "total"


def no_edge_from_start(graph: Graph) -> None:
    graph["edges"][0] = ["embed", "retrieve", "rerank"]


def unknown_key_in_a_stage(graph: Graph) -> None:
    graph["stages"]["rerank"]["top_n"] = 3


def user_prompt_without_its_context(graph: Graph) -> None:
    graph["stages"]["generate"]["user_prompt"] = "Q: {question}"


MISTAKES: dict[str, tuple[Callable[[Graph], None], list[str]]] = {
    "an unknown stage type": (
        unknown_stage_type,
        [
            "stage 'rerank'",
            f"unknown stage type 'rerankk' (expected one of: {STAGE_TYPES})",
        ],
    ),
    "a missing component": (
        missing_component,
        ["stage 'rerank' calls 'rerankr'", f"(expected one of: {COMPONENTS})"],
    ),
    "a component of the wrong type": (
        component_of_the_wrong_type,
        ["stage 'rerank' calls 'llm', a component of type 'llm'", "of type 'reranker'"],
    ),
    "dense retrieval with no embed stage": (
        dense_retrieval_with_no_embed_stage,
        ["stage 'retrieve' reads", "'query_embedding'", "written by no stage"],
    ),
    "an embed stage off one path": (
        embed_stage_off_one_path,
        ["stage 'retrieve' reads", "'query_embedding'", "not on every path"],
    ),
    "an edge to an unknown stage": (
        edge_to_an_unknown_stage,
        ["edge 'generate' -> 'ending'", "'ending'", "(expected END or one of"],
    ),
    "an edge from an unknown stage": (
        edge_from_an_unknown_stage,
        ["edge 'generat' -> 'END'", "(expected START or one of"],
    ),
    "an unregistered edge condition": (
        unregistered_edge_condition,
        ["unknown edge condition 'is_query' (expected one of: 'is_query_mode')"],
    ),
    "a conditional edge without its false target": (
        conditional_edge_without_a_false_target,
        ["conditional edge from 'rerank' on 'is_query_mode' has no false target"],
    ),
    "a conditional edge without its true target": (
        conditional_edge_without_a_true_target,
        ["conditional edge from 'rerank' on 'is_query_mode' has no true target"],
    ),
    "a conditional edge to an unknown stage": (
        conditional_edge_to_an_unknown_stage,
        ["'generat'", "(expected END or one of"],
    ),
    "an edge condition twice on one source": (
        edge_condition_twice_on_one_source,
        ["edge condition 'is_query_mode' is used twice on 'rerank'"],
    ),
    "a stage named like the total": (
        stage_named_like_the_total,
        ["stage 'total' is named like the key of the total time"],
    ),
    "no edge from START": (no_edge_from_start, ["no edge from START"]),
    "an unknown key in a stage": (
        unknown_key_in_a_stage,
        ["stage 'rerank'", "top_n: Extra inputs are not permitted", "'top_k'"],
    ),
    "a user prompt without its context": (
        user_prompt_without_its_context,
        ["stage 'generate'", "lacks the {context} placeholder"],
    ),
}


@pytest.mark.parametrize("mistake", list(MISTAKES))
def test_a_graph_that_cannot_run_is_a_usage_error_before_anything_opens(
    server: FakeServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    mistake: str,
) -> None:
    change, said = MISTAKES[mistake]
    graph = packaged_graph()
    change(graph)

    with pytest.raises(SystemExit) as exited:
        start_on(graph, monkeypatch, tmp_path)

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: rag-serve")
    assert "Traceback" not in err
    for words in said:
        assert words in err
    assert server.apps == []  # the server never started, so nothing opened


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_todays_three_graphs_pass_every_check_and_answer(
    server: FakeServer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    graph = rag_graph(mode).model_dump(exclude_none=True)

    assert start_on(graph, monkeypatch, tmp_path) == 0

    [app] = server.apps
    with TestClient(app) as client:
        response = client.post("/query", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    assert response.json()["answer"]


# --- what only an opened component shows ------------------------------------------


async def start_under_uvicorn(app: Any, caplog: pytest.LogCaptureFixture) -> list[str]:
    """Run the app's start-up as uvicorn does; return what uvicorn logged as errors."""
    config = uvicorn.Config(app, lifespan="on", log_config=None)
    config.load()
    lifespan = LifespanOn(config)

    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        await lifespan.startup()

    assert lifespan.should_exit
    return [record.getMessage() for record in caplog.records]


async def test_a_store_that_ranks_no_text_ends_the_start_up_in_one_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = RagSettings(
        _env_file=None,  # type: ignore[call-arg]
        **{name[4:].lower(): value for name, value in FAKE_BACKENDS.items()},
        graph=rag_graph("lexical"),
    )
    app = create_app(CompositionRoot(settings, vector_store=DenseOnlyStore(["a"])))

    [refusal, exiting] = await start_under_uvicorn(app, caplog)

    assert refusal.startswith("rag-serve: cannot start: stage 'retrieve'")
    assert "retrieval_mode='lexical' needs a store that ranks text" in refusal
    assert "\n" not in refusal
    assert exiting == "Application startup failed. Exiting."


class LostIndex:
    """A lifespan that fails to start, as when a component cannot open."""

    async def __aenter__(self) -> None:
        raise RuntimeError("the index is gone")

    async def __aexit__(self, *exc: object) -> None:
        return None


async def test_any_other_start_up_failure_keeps_its_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = FastAPI(lifespan=lambda app: LostIndex())
    app.add_middleware(GraphRefusalInOneLine)

    [failure, exiting] = await start_under_uvicorn(app, caplog)

    assert failure.startswith("Traceback (most recent call last):")
    assert failure.rstrip().endswith("RuntimeError: the index is gone")
    assert exiting == "Application startup failed. Exiting."
