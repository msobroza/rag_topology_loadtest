"""The stage timings of the workflow app's answers follow the graph that ran.

The app is started from all-fake settings. ``timings_ms`` has one key per stage
of the graph, in the order of its description, 0 for a stage that did not run,
then ``total``; the load-test client's STAGE rows are read from those keys. A
graph whose embed stage has another name shows that name in both.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_graphs import rag_graph
from rag_load_test.composition.composition_root import CompositionRoot
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.graph_description import END, START, GraphDescription
from rag_load_test.entrypoints.rag_serve import create_app
from rag_load_test.loadtest.response_checks import judge_response

QUESTION = "How many paid vacation days do employees accrue each year?"
FAKE_BACKENDS = {
    "embedding_backend": "fake",
    "rerank_backend": "fake",
    "generation_backend": "fake",
    "vector_store_backend": "fake",
}
# Today's response: the four stages in the graph's order, then the total.
TODAY = ["embed", "retrieve", "rerank", "generate", "total"]


def encode_graph() -> GraphDescription:
    """Today's graph with its embed stage named ``encode``; its type stays embed."""
    today = rag_graph()
    encode = today.stages["embed"].model_copy(update={"type": "embed"})
    others = {name: s for name, s in today.stages.items() if name != "embed"}
    edges = [[START, "encode", "retrieve", "rerank"], ["generate", END]]
    return today.model_copy(
        update={"stages": {"encode": encode, **others}, "edges": edges}
    )


def _answer(settings: RagSettings, path: str) -> httpx.Response:
    with TestClient(create_app(CompositionRoot(settings))) as app:
        response = app.post(path, json={"question": QUESTION})
    assert response.status_code == 200, response.text
    return response


def _settings(graph: GraphDescription | None = None) -> RagSettings:
    # A graph handed in whole replaces the workflow file's; none keeps today's.
    graphs = {} if graph is None else {"graph": graph}
    return RagSettings(_env_file=None, **FAKE_BACKENDS, **graphs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "skipped"), [("/query", []), ("/retrieve", ["generate"])]
)
def test_today_s_graph_answers_today_s_timings_key_for_key_in_order(
    path: str, skipped: list[str]
) -> None:
    timings = _answer(_settings(), path).json()["timings_ms"]

    assert list(timings) == TODAY
    assert [stage for stage, ms in timings.items() if ms == 0.0] == skipped


@pytest.mark.parametrize(
    ("path", "rows"),
    [
        (
            "/query",
            ["/query:encode", "/query:retrieve", "/query:rerank", "/query:generate"],
        ),
        ("/retrieve", ["/retrieve:encode", "/retrieve:retrieve", "/retrieve:rerank"]),
    ],
)
def test_a_stage_of_another_name_is_timed_and_made_a_stage_row_by_that_name(
    path: str, rows: list[str]
) -> None:
    response = _answer(_settings(encode_graph()), path)

    assert list(response.json()["timings_ms"]) == ["encode", *TODAY[1:]]
    verdict = judge_response(path, response.status_code, response.text)
    assert [name for name, _ in verdict.stage_rows] == rows
