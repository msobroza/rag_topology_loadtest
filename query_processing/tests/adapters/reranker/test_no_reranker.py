"""The reranker of ``QP_COMPONENTS__RERANKER__BACKEND=none``: retrieval order, no model."""

from __future__ import annotations

from query_processing_core.retrieval.vector_store_record import ScoredVectorStoreRecord

from query_processing.adapters.reranker.none.no_reranker import NoReranker


def scored_records(*ids: str) -> list[ScoredVectorStoreRecord]:
    return [
        ScoredVectorStoreRecord(
            text=f"text of {i}", metadata={"id": i}, retrieval_score=0.5
        )
        for i in ids
    ]


async def test_no_reranker_keeps_order_and_truncates() -> None:
    ranked = await NoReranker().rerank("q", scored_records("a", "b", "c"), top_n=2)
    assert [p.id for p in ranked] == ["a", "b"]
    assert all(p.rerank_score is None for p in ranked)
    assert await NoReranker().ready() == (True, "none")
