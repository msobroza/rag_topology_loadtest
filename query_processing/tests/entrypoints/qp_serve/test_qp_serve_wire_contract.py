"""The wire contract of ``/query`` and ``/retrieve``, written out here in full.

The package, the commands and the environment prefix took new names, but a load
test compares runs from before and after: the request and the response headers
keep theirs. The response body returns the scored vector-store records as they
are, under ``records`` (ADR 0014): a text, the flat metadata with the id and,
when there is one, the chunk, and the two scores.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from doclang_domain.core.adapters.implementations.vector_store import to_vector_store
from doclang_domain.core.chunking import Chunk, ChunkBlock
from doclang_domain.core.document.elements.text import Text
from doclang_domain.core.shared import DocumentHash, DocumentVersionRef, chunk_id_for
from fastapi.testclient import TestClient
from query_processing_core.query_response import QueryResponse
from query_processing_core.retrieval.vector_store_record import (
    VectorStoreRecord,
    from_vector_store,
)

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
    # The scored vector-store records, as they are (ADR 0014).
    "records",
    "timings_ms",
    # Added with the token counts: each stage whose LLM reported both, {} for
    # a body with only a question on /retrieve.
    "tokens",
    # Added with the metadata filter: the conditions applied, [] for a body
    # with only a question.
    "metadata_filter",
}
RECORD_FIELDS = ["text", "metadata", "retrieval_score", "rerank_score"]
# A vector-store record with its id in its metadata, and no chunk.
LEAVE = VectorStoreRecord(
    text="Leave is booked a week ahead.", metadata={"topic": "hr", "id": "leave-1"}
)
# A record made from a chunk holds it under "chunk", as one JSON text: the
# toolkit never reads it, so any text stands for it here (the last test of
# this file seeds doclang's own chunk).
CHUNK_PAYLOAD = '{"id": "doc-leave@v1#chunk-0", "sequence": 0}'
LEAVE_FROM_A_CHUNK = VectorStoreRecord(
    text="Leave is booked a week ahead.",
    metadata={"id": "doc-leave@v1#chunk-0", "chunk": CHUNK_PAYLOAD, "sequence": 0},
)


@pytest.mark.parametrize(
    ("path", "mode"), [("/query", "query"), ("/retrieve", "retrieve")]
)
def test_a_route_answers_with_its_headers_and_the_records_as_they_are(
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
    assert body["records"]
    assert all(list(record) == RECORD_FIELDS for record in body["records"])
    # The fake store's records are the handbook's: the id and the window's
    # index in the metadata, and no chunk.
    for record in body["records"]:
        assert isinstance(record["metadata"]["id"], str)
        assert isinstance(record["metadata"]["sequence"], int)
        assert "chunk" not in record["metadata"]


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


def _records_answered(
    settings_factory: Callable[..., WorkflowSettings],
    path: str,
    record: VectorStoreRecord,
) -> dict[str, Any]:
    composition = CompositionRoot(
        settings_factory(**FAKE_BACKENDS),
        vector_store=_store_holding(record),
        embedder=FakeEmbedder(dim=4),
    )
    with TestClient(create_app(composition)) as client:
        body: dict[str, Any] = client.post(path, json={"question": QUESTION}).json()
    return body


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_a_record_with_no_chunk_comes_back_as_it_is_its_id_in_its_metadata(
    settings_factory: Callable[..., WorkflowSettings], path: str
) -> None:
    body = _records_answered(settings_factory, path, LEAVE)

    [record] = body["records"]
    assert list(record) == RECORD_FIELDS
    assert (record["text"], record["metadata"]) == (LEAVE.text, LEAVE.metadata)
    assert record["rerank_score"] is not None  # reranked, though it has no chunk
    assert (body["answer"] is not None) == (path == "/query")


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_a_record_that_holds_a_chunk_comes_back_with_it_unchanged(
    settings_factory: Callable[..., WorkflowSettings], path: str
) -> None:
    body = _records_answered(settings_factory, path, LEAVE_FROM_A_CHUNK)

    [record] = body["records"]
    assert record["metadata"] == LEAVE_FROM_A_CHUNK.metadata
    assert record["metadata"]["chunk"] == CHUNK_PAYLOAD
    assert record["text"] == LEAVE_FROM_A_CHUNK.text


def handbook_chunk() -> Chunk:
    """A chunk doclang built: one page, one block holding one sentence, with ACLs.

    doclang's ingestion is not installed here: the chunk is built from doclang's
    core, as the core's own test builds one.
    """
    version = DocumentVersionRef(
        document_id="doc-handbook",
        version_no=1,
        document_hash=DocumentHash(value="ab12", hash_function="sha256sum"),
    )
    text = Text.model_validate(
        {"id": "text-1", "content": [{"kind": "span", "text": QUESTION}]}
    )
    return Chunk(
        id=chunk_id_for(version, 0),
        version_ref=version,
        sequence=0,
        chunk_count=1,
        blocks=(ChunkBlock(page_no=1, block_index=0, element=text),),
        page_count=1,
        metadata_core={"tenant": "acme", "acl": {"groups": ["hr", "all"]}},
    )


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_a_chunk_doclang_built_is_rebuilt_from_the_body_a_client_receives(
    settings_factory: Callable[..., WorkflowSettings], path: str
) -> None:
    chunk = handbook_chunk()
    stored = to_vector_store(chunk)

    body = _records_answered(settings_factory, path, stored)

    [record] = body["records"]
    assert record["metadata"] == stored.metadata
    assert from_vector_store(record["metadata"]) == chunk
    [typed] = QueryResponse.model_validate(body).records
    assert from_vector_store(typed) == chunk
    assert record["rerank_score"] is not None
    assert (body["answer"] is not None) == (path == "/query")
