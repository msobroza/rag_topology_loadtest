"""In-memory vector store: the ``fake`` backend.

Vectors are searched by brute-force cosine. Words are ranked by a stand-in, not
BM25: a passage scores the number of distinct query words its text contains.
That is enough to exercise the lexical and hybrid paths without Elasticsearch.

Example:
    >>> store = InMemoryVectorStore()
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ...domain.passages import Passage, ScoredPassage
from ...domain.ports import BaseLexicalSearch, BaseVectorStore
from .fake_models import cosine

_WORD = re.compile(r"\w+")


def distinct_words(text: str) -> set[str]:
    """Split a text into its distinct lower-cased word tokens.

    Args:
        text: Any text.

    Returns:
        The set of its word tokens (runs of letters, digits or _),
        lower-cased.

    Example:
        >>> distinct_words("Vacation, vacation days")
        {'vacation', 'days'}
    """
    return set(_WORD.findall(text.lower()))


class InMemoryVectorStore(BaseVectorStore, BaseLexicalSearch):
    """Passages and vectors in a dict, searched by cosine or by shared words.

    The lexical ranking is a stand-in for a real text ranking such as BM25: the
    score is how many distinct query words the passage text contains.
    """

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

    async def lexical_query(self, text: str, top_k: int) -> list[ScoredPassage]:
        """Find the passages sharing the most distinct words with a text.

        Args:
            text: The text searched for.
            top_k: How many passages to return at most.

        Returns:
            The passages, most shared words first (ties in insertion order),
            with that count as ``retrieval_score``; a passage sharing no word
            is left out.
        """
        wanted = distinct_words(text)
        scored = [
            (len(wanted & distinct_words(p.text)), p) for p, _ in self._rows.values()
        ]
        # sorted() is stable, so equal scores keep the dict's insertion order.
        ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])
        return [
            ScoredPassage(**p.model_dump(), retrieval_score=float(score))
            for score, p in ranked[:top_k]
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
