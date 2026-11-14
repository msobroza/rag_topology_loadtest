"""The ports of the four components: who implements which, and what makes a port."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_load_test.adapters.embedder.base_embedder import BaseEmbedder
from rag_load_test.adapters.embedder.fake.fake_embedder import FakeEmbedder
from rag_load_test.adapters.llm.base_llm import BaseLLM
from rag_load_test.adapters.llm.fake.fake_llm import FakeLLM
from rag_load_test.adapters.reranker.base_reranker import BaseReranker
from rag_load_test.adapters.reranker.fake.fake_reranker import FakeReranker
from rag_load_test.adapters.vector_store.base_vector_store import (
    BaseLexicalSearch,
    BaseVectorStore,
)
from rag_load_test.adapters.vector_store.chroma.chroma_store import ChromaVectorStore
from rag_load_test.adapters.vector_store.elasticsearch.elasticsearch_store import (
    ElasticsearchVectorStore,
)
from rag_load_test.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from rag_load_test.domain.backends import LEXICAL_STORE_BACKENDS
from rag_load_test.domain.passages import ScoredPassage


def test_elasticsearch_and_the_in_memory_store_rank_text_and_chroma_does_not() -> None:
    assert issubclass(ElasticsearchVectorStore, BaseLexicalSearch)
    assert isinstance(InMemoryVectorStore(), BaseLexicalSearch)
    # Open-source Chroma refuses ranked text search (Chroma Cloud only).
    assert not issubclass(ChromaVectorStore, BaseLexicalSearch)
    for store in (ElasticsearchVectorStore, InMemoryVectorStore, ChromaVectorStore):
        assert issubclass(store, BaseVectorStore)


def test_the_lexical_store_backends_are_the_stores_that_rank_text() -> None:
    assert LEXICAL_STORE_BACKENDS == ("elasticsearch", "fake")


def test_fakes_subclass_their_ports() -> None:
    assert isinstance(FakeEmbedder(), BaseEmbedder)
    assert isinstance(FakeReranker(), BaseReranker)
    assert isinstance(InMemoryVectorStore(), BaseVectorStore)
    assert isinstance(FakeLLM(), BaseLLM)
    assert not isinstance(FakeLLM(), BaseVectorStore)


def test_a_port_subclass_missing_a_method_cannot_be_instantiated() -> None:
    class HalfReranker(BaseReranker):
        """Implements ``rerank`` but forgets ``ready``."""

        async def rerank(
            self, query: str, passages: Sequence[ScoredPassage], top_n: int
        ) -> list[ScoredPassage]:
            return []

    with pytest.raises(TypeError, match=r"HalfReranker.*ready"):
        HalfReranker()  # type: ignore[abstract]


def test_a_class_with_the_right_methods_but_no_port_base_is_not_a_port() -> None:
    class DuckEmbedder:
        model_name = "duck"

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return []

        async def ready(self) -> tuple[bool, str]:
            return True, "duck"

    assert not isinstance(DuckEmbedder(), BaseEmbedder)
