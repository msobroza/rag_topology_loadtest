"""In-memory vector store: the ``fake`` backend.

Vectors are searched by brute-force cosine, words through a ``Bm25Index``.

Example:
    >>> store = InMemoryVectorStore(Bm25Index(k1=1.2, b=0.75))
"""

from __future__ import annotations

from collections.abc import Sequence

from ...domain.bm25 import Bm25Index
from ...domain.passages import Passage, ScoredPassage
from ...domain.ports import VectorStorePort
from .fake_models import cosine


class InMemoryVectorStore(VectorStorePort):
    """Passages and vectors in a dict, searched by cosine or by BM25.

    Args:
        lexical_index: The BM25 index every upserted passage is added to.
    """

    def __init__(self, lexical_index: Bm25Index) -> None:
        self._rows: dict[str, tuple[Passage, list[float]]] = {}
        self._lexical_index = lexical_index

    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Store passages with their vectors; a known id is overwritten.

        Args:
            passages: The passages to store.
            embeddings: One vector per passage, in the same order.
        """
        for passage, embedding in zip(passages, embeddings, strict=True):
            self._rows[passage.id] = (passage, list(embedding))
            self._lexical_index.add(passage.id, passage.text)

    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]:
        """Find the passages nearest to a vector by cosine.

        Args:
            embedding: The query vector.
            top_k: How many passages to return at most.

        Returns:
            The nearest passages, nearest first, with the cosine as
            ``retrieval_score``.
        """
        scored = sorted(
            ((cosine(embedding, vec), p) for p, vec in self._rows.values()),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            ScoredPassage(**p.model_dump(), retrieval_score=score)
            for score, p in scored[:top_k]
        ]

    async def lexical_query(self, text: str, top_k: int) -> list[ScoredPassage]:
        """Find the passages whose words best match a text, by BM25.

        Args:
            text: The text searched for.
            top_k: How many passages to return at most.

        Returns:
            The best-matching passages, best first, with the BM25 score as
            ``retrieval_score``.
        """
        return [
            ScoredPassage(**self._rows[pid][0].model_dump(), retrieval_score=score)
            for pid, score in self._lexical_index.search(text, top_k)
        ]

    async def count(self) -> int:
        """Count the stored passages.

        Returns:
            The number of passages in memory.
        """
        return len(self._rows)

    async def ready(self) -> tuple[bool, str]:
        """Report whether any passage is stored.

        Returns:
            Whether the store is non-empty, and how many passages it holds.
        """
        return bool(self._rows), f"{len(self._rows)} passages in memory"
