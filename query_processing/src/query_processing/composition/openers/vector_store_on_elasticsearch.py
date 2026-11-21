"""The vector store on the ``elasticsearch`` backend: kNN, and BM25 for lexical search.

Example:
    >>> store = await open_vector_store_on_elasticsearch(vector_store, shared)
"""

from __future__ import annotations

from ...adapters.vector_store.base_vector_store import BaseVectorStore
from ...adapters.vector_store.elasticsearch.elasticsearch_store import (
    ElasticsearchVectorStore,
)
from ...config.components.component_settings import VectorStoreSettings
from .component_openers import VECTOR_STORE_OPENERS, SharedResources


@VECTOR_STORE_OPENERS.register("elasticsearch")
async def open_vector_store_on_elasticsearch(
    component: VectorStoreSettings, shared: SharedResources
) -> BaseVectorStore:
    """Open the store of the ``elasticsearch`` block, its client closed with the scope.

    Args:
        component: The store's ``elasticsearch`` block.
        shared: The scope, which closes the client, and the key source its
            ``key`` block names, shared with every block naming the same.

    Returns:
        The Elasticsearch store, under the settings' prefilter.

    Raises:
        MissingKeyError: If its key source finds no key, such as an ``env``
            variable unset or empty.

    Example:
        >>> store = await open_vector_store_on_elasticsearch(vector_store, shared)
    """
    block = component.elasticsearch
    store = await ElasticsearchVectorStore.open(
        block.url,
        index=block.index,
        key_source=await shared.http.key_source_of(block.key),
        text_analyzer=block.text_analyzer,
    )
    store.prefilter = tuple(component.prefilter)
    shared.scope.exit_stack("vector_store").push_async_callback(store.aclose)
    return store
