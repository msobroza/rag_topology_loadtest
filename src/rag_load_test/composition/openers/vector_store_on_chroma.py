"""The vector store on the ``chroma`` backend: a Chroma server, or embedded Chroma.

Example:
    >>> store = await open_vector_store_on_chroma(components.vector_store, shared)
"""

from __future__ import annotations

from ...adapters.vector_store.base_vector_store import BaseVectorStore
from ...adapters.vector_store.chroma.chroma_store import ChromaVectorStore
from ...config.component_settings import VectorStoreSettings
from .component_openers import VECTOR_STORE_OPENERS, SharedResources


@VECTOR_STORE_OPENERS.register("chroma")
async def open_vector_store_on_chroma(
    component: VectorStoreSettings, _shared: SharedResources
) -> BaseVectorStore:
    """Open the store of the ``chroma`` block.

    Args:
        component: The store's ``chroma`` block.
        _shared: Unused: Chroma shares nothing with the process.

    Returns:
        The Chroma store.

    Example:
        >>> store = await open_vector_store_on_chroma(components.vector_store, shared)
    """
    block = component.chroma
    return await ChromaVectorStore.open(
        url=block.url, path=block.path, collection=block.collection
    )
