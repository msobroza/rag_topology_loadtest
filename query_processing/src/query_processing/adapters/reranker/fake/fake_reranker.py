"""The ``fake`` reranker: Jaccard overlap of query and record tokens.

Example:
    >>> await FakeReranker().rerank("refund", records, top_n=5)
"""

from __future__ import annotations

from collections.abc import Sequence

from query_processing_core.retrieval.vector_store_record import ScoredVectorStoreRecord

from ....domain.retrieval.rag_result import top_by_score
from ....infrastructure.fake_text import tokenize
from ..base_reranker import BaseReranker


class FakeReranker(BaseReranker):
    """Scores by Jaccard overlap between query and record tokens (0..1)."""

    async def rerank(
        self, query: str, records: Sequence[ScoredVectorStoreRecord], top_n: int
    ) -> list[ScoredVectorStoreRecord]:
        """Score each record by token overlap with the query and keep the best.

        Args:
            query: The question the records are scored against.
            records: The retrieval candidates.
            top_n: How many records to keep.

        Returns:
            At most ``top_n`` records, best first, with their Jaccard score.
        """
        query_tokens = set(tokenize(query))
        scores = [_jaccard(query_tokens, set(tokenize(p.text))) for p in records]
        return top_by_score(records, scores, top_n)

    async def ready(self) -> tuple[bool, str]:
        """Report ready; the fake has nothing to load.

        Returns:
            ``True`` and ``"fake-reranker"``.
        """
        return True, "fake-reranker"


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0
