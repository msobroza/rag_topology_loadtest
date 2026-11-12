"""Deterministic, dependency-free doubles for the model ports (the ``fake`` backends).

The embedder, the reranker and the LLM are here; the vector store's double
is ``in_memory_vector_store.py``, which borrows ``cosine`` from this module.
``FakeEmbedder`` is a hashed bag-of-words embedder, so texts sharing words get
similar vectors and retrieval over the synthetic corpus stays meaningful in
tests and LLM-free load runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Sequence

from ...domain.passages import ScoredPassage, top_by_score
from ...domain.ports import BaseEmbedder, BaseLLM, BaseReranker

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into lower-case alphanumeric tokens.

    Args:
        text: The text to split.

    Returns:
        The runs of ``[a-z0-9]`` in the lower-cased text, in order.

    Example:
        >>> tokenize("Hello, World 42")
        ['hello', 'world', '42']
    """
    return _TOKEN.findall(text.lower())


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the dot product, which is the cosine for unit vectors.

    Args:
        a: A unit vector.
        b: A unit vector of the same length.

    Returns:
        The dot product of ``a`` and ``b``.
    """
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def _token_vector(token: str, dim: int) -> list[float]:
    digest = hashlib.sha256(token.encode()).digest()
    return _normalise([digest[i % len(digest)] / 127.5 - 1.0 for i in range(dim)])


class FakeEmbedder(BaseEmbedder):
    """Hashed bag-of-words embedder: texts sharing words get similar vectors.

    Args:
        dim: Length of every vector.
        model_name: The name ``ready`` reports.
    """

    def __init__(self, dim: int = 16, model_name: str = "fake-embedder") -> None:
        self.dim, self.model_name = dim, model_name

    def embed_text(self, text: str) -> list[float]:
        """Embed one text as the normalised sum of its tokens' hashed vectors.

        Args:
            text: The text to embed; one with no token embeds as ``<empty>``.

        Returns:
            A unit vector of length ``dim``.
        """
        acc = [0.0] * self.dim
        for token in tokenize(text) or ["<empty>"]:
            for i, value in enumerate(_token_vector(token, self.dim)):
                acc[i] += value
        return _normalise(acc)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts with ``embed_text``.

        Args:
            texts: The texts to embed.

        Returns:
            One unit vector per text, in the same order.
        """
        return [self.embed_text(t) for t in texts]

    async def ready(self) -> tuple[bool, str]:
        """Report ready; the fake has nothing to load.

        Returns:
            ``True`` and the model name.
        """
        return True, self.model_name


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


class FakeLLM(BaseLLM):
    """Echoes the last user message; ``latency_ms`` imitates generation time.

    Args:
        latency_ms: Milliseconds each ``generate`` sleeps first; 0 answers at once.
    """

    model_name = "fake-llm"

    def __init__(self, latency_ms: float = 0.0) -> None:
        self.latency_ms, self.calls = latency_ms, 0

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Echo the last user message, after ``latency_ms``; counts the calls.

        Args:
            messages: OpenAI-style messages, each with ``role`` and ``content``.

        Returns:
            ``"[fake-llm] "`` and the first 200 characters of the last user message
            ("" when there is none).
        """
        self.calls += 1
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)
        user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return f"[fake-llm] {user[:200]}"
