"""Deterministic, dependency-free doubles for every port (the ``fake`` backends).

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

from .models import Passage, ScoredPassage, top_by_score

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def _token_vector(token: str, dim: int) -> list[float]:
    digest = hashlib.sha256(token.encode()).digest()
    return _normalise([digest[i % len(digest)] / 127.5 - 1.0 for i in range(dim)])


class FakeEmbedder:
    def __init__(self, dim: int = 16, model_name: str = "fake-embedder") -> None:
        self.dim, self.model_name = dim, model_name

    def embed_text(self, text: str) -> list[float]:
        acc = [0.0] * self.dim
        for token in tokenize(text) or ["<empty>"]:
            for i, value in enumerate(_token_vector(token, self.dim)):
                acc[i] += value
        return _normalise(acc)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    async def ready(self) -> tuple[bool, str]:
        return True, self.model_name


class FakeReranker:
    """Scores by Jaccard overlap between query and passage tokens (0..1)."""

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        query_tokens = set(tokenize(query))
        scores = [_jaccard(query_tokens, set(tokenize(p.text))) for p in passages]
        return top_by_score(passages, scores, top_n)

    async def ready(self) -> tuple[bool, str]:
        return True, "fake-reranker"


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._rows: dict[str, tuple[Passage, list[float]]] = {}

    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        for passage, embedding in zip(passages, embeddings, strict=True):
            self._rows[passage.id] = (passage, list(embedding))

    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]:
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
        return len(self._rows)


class FakeChatModel:
    """Echoes the last user message; ``latency_ms`` imitates generation time."""

    model_name = "fake-llm"

    def __init__(self, latency_ms: float = 0.0) -> None:
        self.latency_ms, self.calls = latency_ms, 0

    async def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)
        user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return f"[fake-llm] {user[:200]}"
