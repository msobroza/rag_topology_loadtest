"""The ``fake`` reranker: Jaccard overlap of query and passage tokens.

Example:
    >>> await FakeReranker().rerank("refund", passages, top_n=5)
"""

from __future__ import annotations

from collections.abc import Sequence

from ....domain.passages import ScoredPassage, top_by_score
from ...fake_text import tokenize
from ..base_reranker import BaseReranker


class FakeReranker(BaseReranker):
    """Scores by Jaccard overlap between query and passage tokens (0..1)."""

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        """Score each passage by token overlap with the query and keep the best.

        Args:
            query: The question the passages are scored against.
            passages: The retrieval candidates.
            top_n: How many passages to keep.

        Returns:
            At most ``top_n`` passages, best first, with their Jaccard score.
        """
        query_tokens = set(tokenize(query))
        scores = [_jaccard(query_tokens, set(tokenize(p.text))) for p in passages]
        return top_by_score(passages, scores, top_n)

    async def ready(self) -> tuple[bool, str]:
        """Report ready; the fake has nothing to load.

        Returns:
            ``True`` and ``"fake-reranker"``.
        """
        return True, "fake-reranker"


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0
