"""The embedder on the ``served`` backend: a model server of ours.

It is called as an ``api`` embedder is, at the endpoint ``served_endpoint``
resolves: the route of the ``served`` block, the standard embedding format,
the alias, and the run token on Domino.

Example:
    >>> embedder = await open_embedder_on_served(embedder_settings, shared)
"""

from __future__ import annotations

from ...adapters.embedder.base_embedder import BaseEmbedder
from ...config.components.component_settings import EmbedderSettings
from .component_openers import EMBEDDER_OPENERS, SharedResources
from .embedder_on_api import open_embedder_at


@EMBEDDER_OPENERS.register("served")
async def open_embedder_on_served(
    component: EmbedderSettings, shared: SharedResources
) -> BaseEmbedder:
    """Open the embedder of the model server the ``served`` block names.

    Args:
        component: The embedder's alias, query prefix, ``served`` and
            ``http`` blocks.
        shared: The shared HTTP client, the default policy, the key sources and
            where the process runs.

    Returns:
        The model server's embedder.

    Example:
        >>> embedder = await open_embedder_on_served(embedder_settings, shared)
    """
    endpoint = shared.http.served_endpoint(component)
    key_source = await shared.http.served_key_source()
    served = component.served
    return await open_embedder_at(endpoint, key_source, served, component, shared)
