"""A ready in-memory store for tests: ``InMemoryVectorStore`` over a fresh BM25 index."""

from __future__ import annotations

from rag_load_test.adapters.fakes.in_memory_vector_store import InMemoryVectorStore
from rag_load_test.domain.bm25 import Bm25Index


def in_memory_store() -> InMemoryVectorStore:
    """An empty store with textbook BM25 parameters (the settings defaults)."""
    return InMemoryVectorStore(Bm25Index(k1=1.2, b=0.75))
