"""The embedder on the ``api`` backend: an endpoint in an embedding API format.

``open_embedder_at`` opens the embedder of any endpoint: the ``served``
backend uses it too, with the endpoint it resolves its own way.

Example:
    >>> embedder = await open_embedder_on_api(embedder_settings, shared)
"""

from __future__ import annotations

from ...adapters.embedder.api.api_embedder import open_api_embedder
from ...adapters.embedder.base_embedder import BaseEmbedder
from ...config.component_settings import EmbedderServedSettings, EmbedderSettings
from ...domain.api_endpoint import ApiEndpoint
from .component_openers import EMBEDDER_OPENERS, SharedResources


@EMBEDDER_OPENERS.register("api")
async def open_embedder_on_api(
    component: EmbedderSettings, shared: SharedResources
) -> BaseEmbedder:
    """Open the embedder of the endpoint the ``api`` block names.

    Args:
        component: The embedder's alias, query prefix, ``api`` and ``http``
            blocks.
        shared: The shared HTTP client, the default policy and the run tokens.

    Returns:
        The endpoint's embedder; an unknown API format is refused here.

    Example:
        >>> embedder = await open_embedder_on_api(embedder_settings, shared)
    """
    api = component.api
    return await open_embedder_at(api.endpoint(component.alias), api, component, shared)


async def open_embedder_at(
    endpoint: ApiEndpoint,
    block: EmbedderServedSettings,
    component: EmbedderSettings,
    shared: SharedResources,
) -> BaseEmbedder:
    """Open the embedder of one endpoint, however it was resolved.

    Args:
        endpoint: The route, its API format, its key and the model name sent.
        block: The block of the component's backend, for the texts per request.
        component: The embedder, for its query prefix and how patiently it is
            called.
        shared: The shared HTTP client, the default policy and the run tokens.

    Returns:
        The endpoint's embedder; an unknown API format is refused here.

    Example:
        >>> embedder = await open_embedder_at(endpoint, component.api, component,
        ...                                   shared)
    """
    http = shared.http
    return open_api_embedder(
        await http.client(),
        endpoint,
        tokens=http.run_tokens,
        policy=http.policy_of(component),
        batch_size=block.batch_size,
        query_prefix=component.query_prefix,
    )
