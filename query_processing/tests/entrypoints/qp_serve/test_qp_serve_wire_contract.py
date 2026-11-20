"""The wire contract of ``/query`` and ``/retrieve`` outlives the toolkit's renames.

The package, the commands and the environment prefix took new names, but a load
test compares runs from before and after: the request, the response headers and
the response bodies keep theirs, written out here in full.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from query_processing_core.retrieval.vector_store_record import VectorStoreRecord

from query_processing.adapters.embedder.fake.fake_embedder import FakeEmbedder
from query_processing.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_serve import create_app
from rag_components import FAKE_BACKENDS

QUESTION = "How many paid vacation days do employees accrue each year?"
BODY_FIELDS = {
    "request_id",
    "topology",
    "run_label",
    "question",
    "mode",
    "answer",
    "passages",
    "timings_ms",
    # Added with the token counts: each stage whose LLM reported both, {} for
    # a body with only a question on /retrieve.
    "tokens",
    # Added with the metadata filter: the conditions applied, [] for a body
    # with only a question.
    "metadata_filter",
}
PASSAGE_FIELDS = {"id", "text", "metadata", "retrieval_score", "rerank_score"}
# A vector-store record with its id in its metadata, and no chunk.
LEAVE = VectorStoreRecord(
    text="Leave is booked a week ahead.", metadata={"topic": "hr", "id": "leave-1"}
)


@pytest.mark.parametrize(
    ("path", "mode"), [("/query", "query"), ("/retrieve", "retrieve")]
)
def test_a_route_answers_with_the_headers_and_body_it_always_had(
    settings_factory: Callable[..., WorkflowSettings], path: str, mode: str
) -> None:
    settings = settings_factory(
        topology_alias="split-all", run_label="contract", **FAKE_BACKENDS
    )

    with TestClient(create_app(CompositionRoot(settings))) as client:
        response = client.post(
            path, json={"question": QUESTION}, headers={"X-Request-ID": "req-1"}
        )

    assert response.status_code == 200, response.text
    assert {
        name: response.headers.get(name)
        for name in ("X-RAG-Topology", "X-RAG-Run-Label", "X-Request-ID")
    } == {
        "X-RAG-Topology": "split-all",
        "X-RAG-Run-Label": "contract",
        "X-Request-ID": "req-1",
    }
    assert [h for h in response.headers if h.lower().startswith("x-qp")] == []
    body = response.json()
    assert set(body) == BODY_FIELDS
    assert (body["request_id"], body["topology"], body["run_label"]) == (
        "req-1",
        "split-all",
        "contract",
    )
    assert (body["question"], body["mode"]) == (QUESTION, mode)
    assert body["metadata_filter"] == []
    assert body["passages"]
    assert all(set(passage) == PASSAGE_FIELDS for passage in body["passages"])


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_a_route_still_takes_its_question_under_the_same_key(
    settings_factory: Callable[..., WorkflowSettings], path: str
) -> None:
    settings = settings_factory(**FAKE_BACKENDS)

    with TestClient(create_app(CompositionRoot(settings))) as client:
        asked = client.post(path, json={"question": QUESTION})
        renamed = client.post(path, json={"query": QUESTION})

    assert (asked.status_code, renamed.status_code) == (200, 422)


def _store_holding(record: VectorStoreRecord) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    asyncio.run(store.upsert([record], [[1.0, 0.0, 0.0, 0.0]]))
    return store


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_an_entry_gives_the_records_id_beside_its_metadata_as_it_always_did(
    settings_factory: Callable[..., WorkflowSettings], path: str
) -> None:
    composition = CompositionRoot(
        settings_factory(**FAKE_BACKENDS),
        vector_store=_store_holding(LEAVE),
        embedder=FakeEmbedder(dim=4),
    )

    with TestClient(create_app(composition)) as client:
        body = client.post(path, json={"question": QUESTION}).json()

    [entry] = body["passages"]
    assert list(entry) == ["id", "text", "metadata", "retrieval_score", "rerank_score"]
    assert (entry["id"], entry["text"], entry["metadata"]) == (
        "leave-1",
        "Leave is booked a week ahead.",
        {"topic": "hr"},  # the id is beside the metadata, not in it
    )
    assert entry["rerank_score"] is not None  # reranked, though it has no chunk
    assert (body["answer"] is not None) == (path == "/query")
