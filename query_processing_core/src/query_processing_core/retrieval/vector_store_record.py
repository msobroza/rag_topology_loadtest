"""The vector-store record, doclang's, and the record with its two scores.

A vector store keeps doclang's ``VectorStoreRecord``: a text and one flat map of
metadata, whose ``id`` it reads, and, for a record made from a chunk, the whole
chunk, which doclang's ``from_vector_store`` rebuilds. This is the one module of
the repository that imports doclang (ADR 0014): the toolkit imports both records
from here.

Example:
    >>> record = ScoredVectorStoreRecord(
    ...     text="Leave is booked a week ahead.",
    ...     metadata={"id": "handbook-7"},
    ...     retrieval_score=0.82,
    ... )
    >>> record.id, record.rerank_score
    ('handbook-7', None)
"""

from __future__ import annotations

from doclang_domain.core.adapters.implementations.vector_store import (
    VectorStoreRecord,
)

__all__ = ["ScoredVectorStoreRecord", "VectorStoreRecord"]


class ScoredVectorStoreRecord(VectorStoreRecord):
    """A vector-store record with the score its search gave it, and its rerank's.

    A record is frozen and refuses an unknown key, so the scores cannot sit
    beside it: they are fields of this subclass, which keeps both rules and the
    id read from the metadata.

    Attributes:
        retrieval_score: The score the search gave the record.
        rerank_score: The score a reranker gave it; ``None`` until one has.

    Example:
        >>> ScoredVectorStoreRecord(
        ...     text="t", metadata={"id": "a"}, retrieval_score=1.0, rerank_score=2.5
        ... ).rerank_score
        2.5
    """

    retrieval_score: float
    rerank_score: float | None = None
