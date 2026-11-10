"""In-memory vector store: the ``fake`` backend, searched by brute-force cosine."""

from __future__ import annotations

from collections.abc import Sequence

from ...domain.passages import Passage, ScoredPassage
from ...domain.ports import VectorStorePort
from .fake_models import cosine


class InMemoryVectorStore(VectorStorePort):
    """Passages and vectors in a dict, searched by brute-force cosine."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[Passage, list[float]]] = {}

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
