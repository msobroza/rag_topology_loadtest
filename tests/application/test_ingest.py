"""Tests for ``application.ingest``: every passage is embedded and upserted, in batches."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from rag_load_test.adapters.embedder.fake.fake_embedder import FakeEmbedder
from rag_load_test.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from rag_load_test.application.ingest import ingest
from rag_load_test.domain.corpus import (
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
    """FakeEmbedder recording the size of every ``embed`` batch."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return await super().embed(texts)


async def test_ingest_upserts_every_passage_in_batches() -> None:
    passages = [p for doc in synthetic_corpus(7) for p in chunk(doc)]
    assert len(passages) == 7  # each synthetic document fits one chunk
    embedder, store = SpyEmbedder(), InMemoryVectorStore()
    assert await ingest(passages, embedder, store, batch_size=3) == 7
    assert await store.count() == 7
    assert embedder.batch_sizes == [3, 3, 1]
    hits = await store.query(embedder.embed_text(passages[4].text), top_k=1)
    assert hits[0].id == passages[4].id
