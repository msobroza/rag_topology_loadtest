"""Reciprocal rank fusion: one ranking out of several.

Hybrid retrieval runs a dense and a lexical query and fuses them here, so it
behaves the same on every vector store and needs nothing from the store itself.

Example:
    >>> reciprocal_rank_fusion([dense_hits, lexical_hits], k=60, top_k=20)
"""

from __future__ import annotations

from collections.abc import Sequence

from query_processing_core.retrieval.vector_store_record import ScoredVectorStoreRecord

from .rag_result import required_record_id


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredVectorStoreRecord]], *, k: int, top_k: int
) -> list[ScoredVectorStoreRecord]:
    """Fuse rankings by summing ``1 / (k + rank)`` per vector-store record.

    Only ranks count, so the rankings' own scores need not be comparable (a
    cosine and a BM25 score are not).

    Args:
        rankings: The rankings to fuse, each best first.
        k: The rank constant; higher flattens the gap between ranks.
        top_k: How many vector-store records to keep.

    Returns:
        At most ``top_k`` records, best first, each with its fused score as
        ``retrieval_score``. Records with equal scores keep the order in which
        the rankings first named them.

    Raises:
        ValueError: A record holds no id (``required_record_id``).

    Example:
        >>> [r.id for r in reciprocal_rank_fusion([[a, b], [b, c]], k=60, top_k=2)]
        ['b', 'a']
    """
    fused: dict[str, float] = {}
    first_seen: dict[str, ScoredVectorStoreRecord] = {}
    for ranking in rankings:
        for rank, record in enumerate(ranking, start=1):
            key = required_record_id(record)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, record)
    best = sorted(fused, key=lambda record_id: fused[record_id], reverse=True)
    return [
        first_seen[record_id].model_copy(update={"retrieval_score": fused[record_id]})
        for record_id in best[:top_k]
    ]
