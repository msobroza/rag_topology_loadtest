"""The reranker on the ``api`` backend: an endpoint in a rerank API format.

``open_reranker_at`` opens the reranker of any endpoint: the ``served``
backend uses it too, with the endpoint it resolves its own way.

Example:
    >>> reranker = await open_reranker_on_api(reranker_settings, shared)
"""

from __future__ import annotations

from ...adapters.key_source.base_key_source import BaseKeySource
from ...adapters.reranker.api.api_reranker import open_api_reranker
from ...adapters.reranker.base_reranker import BaseReranker
from ...config.components.component_settings import RerankerSettings
from ...domain.endpoints.api_endpoint import ApiEndpoint
from .component_openers import RERANKER_OPENERS, SharedResources


@RERANKER_OPENERS.register("api")
async def open_reranker_on_api(
    component: RerankerSettings, shared: SharedResources
) -> BaseReranker:
    """Open the reranker of the endpoint the ``api`` block names.

    Args:
        component: The reranker's alias, ``api`` and ``http`` blocks.
        shared: The shared HTTP client, the default policy and the key sources.

    Returns:
        The endpoint's reranker; an unknown API format is refused here.

    Example:
        >>> reranker = await open_reranker_on_api(reranker_settings, shared)
    """
    endpoint = component.api.endpoint(component.alias)
    key_source = await shared.http.key_source_of(component.api.key)
    return await open_reranker_at(endpoint, key_source, component, shared)


async def open_reranker_at(
    endpoint: ApiEndpoint,
    key_source: BaseKeySource | None,
    component: RerankerSettings,
    shared: SharedResources,
) -> BaseReranker:
    """Open the reranker of one endpoint, however it was resolved.

    Args:
        endpoint: The route, its API format and the model name sent.
        key_source: Where its API key comes from; None sends no key.
        component: The reranker, for how patiently it is called.
        shared: The shared HTTP client, the default policy and the key sources.

    Returns:
        The endpoint's reranker; an unknown API format is refused here.

    Example:
        >>> reranker = await open_reranker_at(endpoint, None, component, shared)
    """
    http = shared.http
    policy = http.policy_of(component)
    return open_api_reranker(
        await http.client(), endpoint, key_source=key_source, policy=policy
    )
