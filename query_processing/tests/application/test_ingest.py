"""Tests for ``application.ingest``: every passage is embedded and upserted, in batches."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from query_processing.adapters.embedder.fake.fake_embedder import FakeEmbedder
from query_processing.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from query_processing.application.ingest import ingest
from query_processing.domain.retrieval.corpus import (
    TOPICS,
    SourceDocument,
    chunk,
    synthetic_corpus,
)

SLUGS = list(TOPICS)


def _doc(words: int, **metadata: str | int) -> SourceDocument:
    """Text ``w0 w1 ... w<words-1>`` so chunk boundaries are checkable."""
    text = " ".join(f"w{i}" for i in range(words))
    return SourceDocument(id="doc-0007", title="Handbook", text=text, metadata=metadata)


def _jsonl(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _record(**fields: object) -> str:
    return json.dumps({"id": "a", "title": "A", "text": "alpha", **fields})


# --- ingest ---


class SpyEmbedder(FakeEmbedder):
    """FakeEmbedder recording every batch of passages and every query."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []
        self.queries: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return await super().embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return await super().embed_query(text)


async def test_ingest_upserts_every_passage_in_batches() -> None:
    passages = [p for doc in synthetic_corpus(7) for p in chunk(doc)]
    assert len(passages) == 7  # each synthetic document fits one chunk
    embedder, store = SpyEmbedder(), InMemoryVectorStore()
    assert await ingest(passages, embedder, store, batch_size=3) == 7
    assert await store.count() == 7
    assert embedder.batch_sizes == [3, 3, 1]
    assert embedder.queries == []  # passages, never embedded as a query
    hits = await store.query(embedder.embed_text(passages[4].text), top_k=1)
    assert hits[0].id == passages[4].id
