"""Builds the vector store from the settings: ``RAG_VECTOR_STORE_BACKEND`` picks which.

Example:
    >>> store = await VectorStoreFactory(settings, scope, embedders.open).open()
"""

from __future__ import annotations

from collections.abc import Mapping

from ..adapters.embedder.base_embedder import BaseEmbedder
from ..adapters.vector_store.base_vector_store import BaseVectorStore
from ..adapters.vector_store.chroma.chroma_store import ChromaVectorStore
from ..adapters.vector_store.elasticsearch.elasticsearch_store import (
    ElasticsearchVectorStore,
)
from ..adapters.vector_store.fake.in_memory_vector_store import InMemoryVectorStore
from ..application.ingest import ingest
from ..config.rag_settings import RagSettings
from ..domain.corpus import chunk, synthetic_corpus
from .base_port_factory import BasePortFactory
from .resource_scope import Opener, ResourceScope


class VectorStoreFactory(BasePortFactory[BaseVectorStore]):
    """The vector store of each Backend: ``elasticsearch``, ``chroma`` or ``fake``.

    Args:
        settings: The store Backend and the Elasticsearch and Chroma settings.
        scope: Opens the store once and closes the Elasticsearch client.
        open_embedder: Opens the embedder the ``fake`` store is seeded with.
        supplied: A ready-made store, used as given; None to build one.

    Example:
        >>> await VectorStoreFactory(settings, scope, embedders.open).open()
    """

    port = "vector_store"

    def __init__(
        self,
        settings: RagSettings,
        scope: ResourceScope,
        open_embedder: Opener[BaseEmbedder],
        supplied: BaseVectorStore | None = None,
    ) -> None:
        super().__init__(settings, scope, supplied)
        self._open_embedder = open_embedder

    def backend(self) -> str:
        """Return ``RAG_VECTOR_STORE_BACKEND``.

        Returns:
            ``elasticsearch``, ``chroma`` or ``fake``.
        """
        return self._settings.vector_store_backend

    def openers(self) -> Mapping[str, Opener[BaseVectorStore]]:
        """Return what opens the vector store of each Backend.

        Returns:
            One entry per vector store Backend value.
        """
        return {
            "elasticsearch": self._elasticsearch,
            "chroma": self._chroma,
            "fake": self._seeded_in_memory,
        }

    async def _elasticsearch(self) -> BaseVectorStore:
        settings = self._settings
        store = ElasticsearchVectorStore.open(
            settings.es_url,
            index=settings.es_index,
            api_key=settings.es_api_key,
            text_analyzer=settings.es_text_analyzer,
        )
        self._scope.exit_stack("vector_store").push_async_callback(store.aclose)
        return store

    async def _chroma(self) -> BaseVectorStore:
        settings = self._settings
        return await ChromaVectorStore.open(
            url=settings.chroma_url,
            path=settings.chroma_path,
            collection=settings.chroma_collection,
        )

    # Seeded with the synthetic corpus, so a fake-store app answers like an
    # ingested one.
    async def _seeded_in_memory(self) -> BaseVectorStore:
        store = InMemoryVectorStore()
        documents = synthetic_corpus(self._settings.fake_store_seed_documents)
        passages = [passage for document in documents for passage in chunk(document)]
        await ingest(passages, await self._open_embedder(), store)
        return store
