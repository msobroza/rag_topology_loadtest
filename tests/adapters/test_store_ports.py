"""Which stores rank text: the ``BaseLexicalSearch`` port each store implements."""

from __future__ import annotations

from rag_load_test.adapters.fakes.in_memory_vector_store import InMemoryVectorStore
from rag_load_test.adapters.vector_store.chroma_store import ChromaVectorStore
from rag_load_test.adapters.vector_store.elasticsearch_store import (
    ElasticsearchVectorStore,
)
from rag_load_test.domain.ports import BaseLexicalSearch, BaseVectorStore
from rag_load_test.domain.topology import LEXICAL_STORE_BACKENDS


def test_elasticsearch_and_the_in_memory_store_rank_text_and_chroma_does_not() -> None:
    assert issubclass(ElasticsearchVectorStore, BaseLexicalSearch)
    assert isinstance(InMemoryVectorStore(), BaseLexicalSearch)
    # Open-source Chroma refuses ranked text search (Chroma Cloud only).
    assert not issubclass(ChromaVectorStore, BaseLexicalSearch)
    for store in (ElasticsearchVectorStore, InMemoryVectorStore, ChromaVectorStore):
        assert issubclass(store, BaseVectorStore)


def test_the_lexical_store_backends_are_the_stores_that_rank_text() -> None:
    assert LEXICAL_STORE_BACKENDS == ("elasticsearch", "fake")
