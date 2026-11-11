"""BM25 lexical ranking over passages held in memory, with an inverted index.

For the stores that have no ranked text search of their own (the in-memory store
and Chroma, whose open-source build only filters by substring). Elasticsearch
ranks with its own BM25 and never uses this.

Example:
    >>> index = Bm25Index(k1=1.2, b=0.75)
    >>> index.add("p1", "Employees accrue 25 vacation days")
    >>> index.search("vacation days", top_k=5)
    [('p1', 0.57...)]
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Letters and digits of any script, so accented words stay whole ("congés").
_WORD = re.compile(r"[^\W_]+")


def tokenize_words(text: str) -> list[str]:
    """Split text into lower-case words.

    Args:
        text: The text to split.

    Returns:
        The runs of letters and digits in the lower-cased text, in order.

    Example:
        >>> tokenize_words("Congés payés: 25 jours")
        ['congés', 'payés', '25', 'jours']
    """
    return _WORD.findall(text.lower())


class Bm25Index:
    """An inverted index scored with Okapi BM25.

    Args:
        k1: Term-frequency saturation; higher lets repeated terms count for more.
        b: Length normalisation, 0 (none) to 1 (full).

    Example:
        >>> index = Bm25Index(k1=1.2, b=0.75)
        >>> index.add("p1", "vacation policy")
        >>> [passage_id for passage_id, _ in index.search("vacation", top_k=1)]
        ['p1']
    """

    def __init__(self, k1: float, b: float) -> None:
        self.k1, self.b = k1, b
        self._terms: dict[str, Counter[str]] = {}  # passage id -> term counts
        self._postings: dict[str, dict[str, int]] = {}  # term -> passage id -> count
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._terms)

    def add(self, passage_id: str, text: str) -> None:
        """Index a passage; a known id is replaced.

        Args:
            passage_id: The passage's id.
            text: The passage's text.
        """
        self._remove(passage_id)
        counts = Counter(tokenize_words(text))
        self._terms[passage_id] = counts
        self._total_length += counts.total()
        for term, count in counts.items():
            self._postings.setdefault(term, {})[passage_id] = count

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Rank the passages sharing a word with the query.

        Args:
            query: The text searched for.
            top_k: How many passages to return at most.

        Returns:
            ``(passage id, score)`` pairs, best first; a passage sharing no word
            with the query is left out.
        """
        scores: dict[str, float] = {}
        for term in set(tokenize_words(query)):
            postings = self._postings.get(term, {})
            idf = self._idf(len(postings))
            for passage_id, count in postings.items():
                gain = idf * self._saturated(count, passage_id)
                scores[passage_id] = scores.get(passage_id, 0.0) + gain
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def _remove(self, passage_id: str) -> None:
        counts = self._terms.pop(passage_id, None)
        if counts is None:
            return
        self._total_length -= counts.total()
        for term in counts:
            del self._postings[term][passage_id]

    def _idf(self, document_frequency: int) -> float:
        rarity = (len(self) - document_frequency + 0.5) / (document_frequency + 0.5)
        return math.log(1.0 + rarity)

    def _saturated(self, count: int, passage_id: str) -> float:
        average = self._total_length / len(self)
        length = self._terms[passage_id].total() / average if average else 0.0
        norm = self.k1 * (1.0 - self.b + self.b * length)
        return count * (self.k1 + 1.0) / (count + norm)
