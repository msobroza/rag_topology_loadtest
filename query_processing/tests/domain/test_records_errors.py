"""The dependency-free core: ``top_by_score``, a record's id and the dependency error."""

from __future__ import annotations

import pytest
from query_processing_core.retrieval.vector_store_record import (
    ScoredVectorStoreRecord,
    VectorStoreRecord,
)

from query_processing.domain.errors import RagDependencyError, describe_error
from query_processing.domain.retrieval.rag_result import (
    required_record_id,
    top_by_score,
)


def _scored(
    pid: str, text: str = "", retrieval_score: float = 0.5
) -> ScoredVectorStoreRecord:
    return ScoredVectorStoreRecord(
        text=text or pid, metadata={"id": pid}, retrieval_score=retrieval_score
    )


def test_top_by_score_orders_desc_truncates_and_stamps_float_rerank_score() -> None:
    records = [_scored("a", retrieval_score=0.1), _scored("b"), _scored("c")]
    kept = top_by_score(records, scores=[1, 3, 2], top_n=2)  # ints on purpose

    assert [p.id for p in kept] == ["b", "c"]
    assert [p.rerank_score for p in kept] == [3.0, 2.0]
    assert all(type(p.rerank_score) is float for p in kept)
    assert [p.retrieval_score for p in kept] == [0.5, 0.5]
    assert records[0].rerank_score is None  # inputs are copied, not mutated


def test_rag_dependency_error_message_carries_component_kind_and_target() -> None:
    err = RagDependencyError("reranker", "HTTP 503", target="http://ovms:8001")

    assert err.kind == "unavailable"
    assert str(err) == "reranker unavailable: HTTP 503 (target=http://ovms:8001)"
    assert "timeout" in str(RagDependencyError("llm", "slow", kind="timeout"))


def test_describe_error_names_the_type_and_the_message_when_there_is_one() -> None:
    described = describe_error(TimeoutError("read timed out"))
    assert described == "TimeoutError: read timed out"
    assert describe_error(TimeoutError()) == "TimeoutError"


def test_a_record_with_an_id_in_its_metadata_gives_it_as_text() -> None:
    record = VectorStoreRecord(text="t", metadata={"id": "handbook-7", "page": 3})
    assert required_record_id(record) == "handbook-7"


def test_a_record_with_no_id_is_refused_naming_its_text_and_the_shape() -> None:
    long_text = "Leave is booked a week ahead. " * 10
    record = VectorStoreRecord(text=long_text, metadata={"topic": "hr"})

    with pytest.raises(ValueError, match="holds no id") as refused:
        required_record_id(record)

    message = str(refused.value)
    assert "'Leave is booked a week ahead. Leave is " in message  # its start
    assert long_text not in message  # not the whole text
    assert 'a text id under metadata["id"]' in message
    assert "['topic']" in message  # the keys it holds instead
