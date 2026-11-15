"""The ``fake`` reranker: token overlap with the query."""

from __future__ import annotations

from rag_load_test.adapters.reranker.fake.fake_reranker import FakeReranker
from rag_load_test.domain.passages import ScoredPassage


def _scored(pid: str, text: str = "", retrieval_score: float = 0.5) -> ScoredPassage:
    return ScoredPassage(id=pid, text=text or pid, retrieval_score=retrieval_score)


async def test_fake_reranker_scores_by_overlap_and_truncates() -> None:
    passages = [
        _scored("off", "gardening tips for spring"),
        _scored("hit", "the elasticsearch cluster shards"),
        _scored("near", "cluster of stars"),
    ]
    kept = await FakeReranker().rerank("elasticsearch cluster", passages, top_n=2)

    assert [p.id for p in kept] == ["hit", "near"]
    assert kept[0].rerank_score is not None
    assert kept[0].rerank_score > 0
    assert kept[0].rerank_score > (kept[1].rerank_score or 0)
