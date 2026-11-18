"""The reranker of ``QP_COMPONENTS__RERANKER__BACKEND=none``: retrieval order, no model."""

from __future__ import annotations

from query_processing.adapters.reranker.none.no_reranker import NoReranker
from query_processing.domain.retrieval.passages import ScoredPassage


def scored_passages(*ids: str) -> list[ScoredPassage]:
    return [ScoredPassage(id=i, text=f"text of {i}", retrieval_score=0.5) for i in ids]


async def test_no_reranker_keeps_order_and_truncates() -> None:
    ranked = await NoReranker().rerank("q", scored_passages("a", "b", "c"), top_n=2)
    assert [p.id for p in ranked] == ["a", "b"]
    assert all(p.rerank_score is None for p in ranked)
    assert await NoReranker().ready() == (True, "none")
