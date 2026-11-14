"""Reciprocal rank fusion: one ranking out of several.

Hybrid retrieval runs a dense and a lexical query and fuses them here, so it
behaves the same on every vector store and needs nothing from the store itself.

Example:
    >>> reciprocal_rank_fusion([dense_hits, lexical_hits], k=60, top_k=20)
"""

from __future__ import annotations

from collections.abc import Sequence

from .passages import ScoredPassage


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredPassage]], *, k: int, top_k: int
) -> list[ScoredPassage]:
    """Fuse rankings by summing ``1 / (k + rank)`` per passage.

    Only ranks count, so the rankings' own scores need not be comparable (a
    cosine and a BM25 score are not).

    Args:
        rankings: The rankings to fuse, each best first.
        k: The rank constant; higher flattens the gap between ranks.
        top_k: How many passages to keep.

    Returns:
        At most ``top_k`` passages, best first, each with its fused score as
        ``retrieval_score``. Passages with equal scores keep the order in which
        the rankings first named them.

    Example:
        >>> [p.id for p in reciprocal_rank_fusion([[a, b], [b, c]], k=60, top_k=2)]
        ['b', 'a']
    """
    fused: dict[str, float] = {}
    first_seen: dict[str, ScoredPassage] = {}
    for ranking in rankings:
        for rank, passage in enumerate(ranking, start=1):
            fused[passage.id] = fused.get(passage.id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(passage.id, passage)
    best = sorted(fused, key=lambda passage_id: fused[passage_id], reverse=True)
    return [
        first_seen[passage_id].model_copy(update={"retrieval_score": fused[passage_id]})
        for passage_id in best[:top_k]
    ]
