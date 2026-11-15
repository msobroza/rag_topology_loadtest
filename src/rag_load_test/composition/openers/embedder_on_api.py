"""The embedder on the ``api`` backend: an endpoint in an embedding API format.

Example:
    >>> embedder = await open_embedder_on_api(components.embedder, shared)
"""

from __future__ import annotations

from ...adapters.embedder.api.api_embedder import open_api_embedder
from ...adapters.embedder.base_embedder import BaseEmbedder
from ...config.component_settings import EmbedderSettings
from .component_openers import EMBEDDER_OPENERS, SharedResources


@EMBEDDER_OPENERS.register("api")
async def open_embedder_on_api(
    component: EmbedderSettings, shared: SharedResources
) -> BaseEmbedder:
    """Open the embedder of the endpoint the ``api`` block names.

    Args:
        component: The embedder's query prefix and ``api`` block.
        shared: The shared HTTP client, its policy and the run tokens.

    Returns:
        The endpoint's embedder; an unknown API format is refused here.

    Example:
        >>> embedder = await open_embedder_on_api(components.embedder, shared)
    """
    http, api = shared.http, component.api
    return open_api_embedder(
        await http.client(),
        api.endpoint(),
        tokens=http.run_tokens,
        policy=http.policy,
        batch_size=api.batch_size,
        query_prefix=component.query_prefix,
    )
