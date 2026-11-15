"""Each retrieval mode, seen from outside the workflow app: what it answers and asks.

The app is started from all-fake settings with today's graph in a retrieval
mode; the embedder is a recording double, so a test reads whether a vector was
computed. A lexical graph has no embed stage: nothing is embedded, and
``timings_ms``, which follows the graph, has no ``embed`` key.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_serve import create_app
from rag_components import FAKE_BACKENDS
from rag_graphs import rag_graph
from workflow_fakes import DenseOnlyStore, RecordingEmbedder

QUESTION = "How many paid vacation days do employees accrue each year?"
# The stages of each mode's graph in order, then the total; lexical has no embed.
TIMING_KEYS = {
    "dense": ["embed", "retrieve", "rerank", "generate", "total"],
    "lexical": ["retrieve", "rerank", "generate", "total"],
    "hybrid": ["embed", "retrieve", "rerank", "generate", "total"],
}
SettingsFactory = Callable[..., WorkflowSettings]


@pytest.mark.parametrize(
    ("mode", "embedded"),
    [("dense", [QUESTION]), ("lexical", []), ("hybrid", [QUESTION])],
)
@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_each_retrieval_mode_answers_and_embeds_only_when_it_needs_a_vector(
    settings_factory: SettingsFactory, mode: str, embedded: list[str], path: str
) -> None:
    settings = settings_factory(**FAKE_BACKENDS, graph=rag_graph(mode))
    embedder = RecordingEmbedder()  # seeds the store too, through embed_documents

    with TestClient(create_app(CompositionRoot(settings, embedder=embedder))) as app:
        response = app.post(path, json={"question": QUESTION})

    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body["timings_ms"]) == TIMING_KEYS[mode]
    assert len(body["passages"]) == 10  # the fixture's rerank stage's top_k
    assert embedder.queries == embedded
    assert (body["timings_ms"].get("embed", 0.0) > 0) == bool(embedded)
    assert (body["answer"] is not None) == (path == "/query")


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_the_fake_store_ranks_the_vacation_policy_first_in_each_mode(
    settings_factory: SettingsFactory, mode: str
) -> None:
    settings = settings_factory(**FAKE_BACKENDS, graph=rag_graph(mode))

    with TestClient(create_app(CompositionRoot(settings))) as app:
        passages = app.post("/retrieve", json={"question": QUESTION}).json()["passages"]

    assert passages[0]["metadata"]["topic"] == "vacation-policy"


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_a_store_that_ranks_no_text_stops_a_text_ranking_app_at_start_up(
    settings_factory: SettingsFactory, mode: str
) -> None:
    settings = settings_factory(**FAKE_BACKENDS, graph=rag_graph(mode))
    app = create_app(CompositionRoot(settings, vector_store=DenseOnlyStore(["a"])))

    with (
        pytest.raises(ValueError, match=f"retrieval_mode='{mode}'.*ranks no text"),
        TestClient(app),
    ):
        pass
