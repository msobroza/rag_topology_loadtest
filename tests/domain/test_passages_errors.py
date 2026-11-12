"""The dependency-free core: ``top_by_score`` and the dependency error."""

from __future__ import annotations

from rag_load_test.domain.errors import RagDependencyError, describe_error
from rag_load_test.domain.passages import ScoredPassage, top_by_score


def _scored(pid: str, text: str = "", retrieval_score: float = 0.5) -> ScoredPassage:
    return ScoredPassage(id=pid, text=text or pid, retrieval_score=retrieval_score)


def test_top_by_score_orders_desc_truncates_and_stamps_float_rerank_score() -> None:
    passages = [_scored("a", retrieval_score=0.1), _scored("b"), _scored("c")]
    kept = top_by_score(passages, scores=[1, 3, 2], top_n=2)  # ints on purpose

    assert [p.id for p in kept] == ["b", "c"]
    assert [p.rerank_score for p in kept] == [3.0, 2.0]
    assert all(type(p.rerank_score) is float for p in kept)
    assert [p.retrieval_score for p in kept] == [0.5, 0.5]
    assert passages[0].rerank_score is None  # inputs are copied, not mutated


def test_rag_dependency_error_message_carries_component_kind_and_target() -> None:
    err = RagDependencyError("reranker", "HTTP 503", target="http://ovms:8001")

    assert err.kind == "unavailable"
    assert str(err) == "reranker unavailable: HTTP 503 (target=http://ovms:8001)"
    assert "timeout" in str(RagDependencyError("llm", "slow", kind="timeout"))


def test_describe_error_names_the_type_and_the_message_when_there_is_one() -> None:
    described = describe_error(TimeoutError("read timed out"))
    assert described == "TimeoutError: read timed out"
    assert describe_error(TimeoutError()) == "TimeoutError"
