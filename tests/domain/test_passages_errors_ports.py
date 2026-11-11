"""The dependency-free core: ``top_by_score``, the dependency error and the port base classes."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_load_test.adapters.fakes.fake_models import (
    FakeEmbedder,
    FakeLLM,
    FakeReranker,
)
from rag_load_test.adapters.fakes.in_memory_vector_store import InMemoryVectorStore
from rag_load_test.domain.errors import RagDependencyError, describe_error
from rag_load_test.domain.passages import ScoredPassage, top_by_score
from rag_load_test.domain.ports import (
    BaseEmbedder,
    BaseLLM,
    BaseReranker,
    BaseVectorStore,
)


def _scored(pid: str, text: str = "", retrieval_score: float = 0.5) -> ScoredPassage:
    return ScoredPassage(id=pid, text=text or pid, retrieval_score=retrieval_score)


# --- models ------------------------------------------------------------------


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


def test_fakes_subclass_their_ports() -> None:
    assert isinstance(FakeEmbedder(), BaseEmbedder)
    assert isinstance(FakeReranker(), BaseReranker)
    assert isinstance(InMemoryVectorStore(), BaseVectorStore)
    assert isinstance(FakeLLM(), BaseLLM)
    assert not isinstance(FakeLLM(), BaseVectorStore)


def test_a_port_subclass_missing_a_method_cannot_be_instantiated() -> None:
    class HalfReranker(BaseReranker):
        """Implements ``rerank`` but forgets ``ready``."""

        async def rerank(
            self, query: str, passages: Sequence[ScoredPassage], top_n: int
        ) -> list[ScoredPassage]:
            return []

    with pytest.raises(TypeError, match=r"HalfReranker.*ready"):
        HalfReranker()  # type: ignore[abstract]


def test_a_class_with_the_right_methods_but_no_port_base_is_not_a_port() -> None:
    class DuckEmbedder:
        model_name = "duck"

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return []

        async def ready(self) -> tuple[bool, str]:
            return True, "duck"

    assert not isinstance(DuckEmbedder(), BaseEmbedder)
