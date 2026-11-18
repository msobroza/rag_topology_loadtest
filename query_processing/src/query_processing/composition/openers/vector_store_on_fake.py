"""The vector store on the ``fake`` backend: in memory, seeded through the embedder.

Example:
    >>> store = await open_vector_store_on_fake(vector_store, shared)
"""

from __future__ import annotations

from ...adapters.vector_store.base_vector_store import BaseVectorStore
from ...adapters.vector_store.fake.in_memory_vector_store import InMemoryVectorStore
from ...application.ingest import ingest
from ...config.components.component_settings import VectorStoreSettings
from ...domain.retrieval.corpus import chunk, synthetic_corpus
from .component_openers import VECTOR_STORE_OPENERS, SharedResources


@VECTOR_STORE_OPENERS.register("fake")
async def open_vector_store_on_fake(
    component: VectorStoreSettings, shared: SharedResources
) -> BaseVectorStore:
    """Open an in-memory store seeded with the synthetic corpus.

    Seeded, so a fake-store app answers like an ingested one; through the
    process's embedder, so the store holds the vectors its queries are given.

    Args:
        component: The store's ``fake`` block: how many documents seed it.
        shared: Opens the process's embedder.

    Returns:
        The seeded in-memory store, under the settings' prefilter.

    Example:
        >>> store = await open_vector_store_on_fake(vector_store, shared)
    """
    store = InMemoryVectorStore()
    documents = synthetic_corpus(component.fake.seed_documents)
    passages = [passage for document in documents for passage in chunk(document)]
    await ingest(passages, await shared.open_embedder(), store)
    store.prefilter = tuple(component.prefilter)
    return store
