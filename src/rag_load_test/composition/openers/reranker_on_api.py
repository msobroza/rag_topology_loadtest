"""The reranker on the ``api`` backend: an endpoint in a rerank API format.

``open_reranker_at`` opens the reranker of any endpoint: the ``served``
backend uses it too, with the endpoint it resolves its own way.

Example:
    >>> reranker = await open_reranker_on_api(reranker_settings, shared)
"""

from __future__ import annotations

from ...adapters.reranker.api.api_reranker import open_api_reranker
from ...adapters.reranker.base_reranker import BaseReranker
from ...config.component_settings import RerankerSettings
from ...domain.api_endpoint import ApiEndpoint
from .component_openers import RERANKER_OPENERS, SharedResources


@RERANKER_OPENERS.register("api")
async def open_reranker_on_api(
    component: RerankerSettings, shared: SharedResources
) -> BaseReranker:
    """Open the reranker of the endpoint the ``api`` block names.

    Args:
        component: The reranker's alias and ``api`` block.
        shared: The shared HTTP client, its policy and the run tokens.

    Returns:
        The endpoint's reranker; an unknown API format is refused here.

    Example:
        >>> reranker = await open_reranker_on_api(reranker_settings, shared)
    """
    return await open_reranker_at(component.api.endpoint(component.alias), shared)


async def open_reranker_at(
    endpoint: ApiEndpoint, shared: SharedResources
) -> BaseReranker:
    """Open the reranker of one endpoint, however it was resolved.

    Args:
        endpoint: The route, its API format, its key and the model name sent.
        shared: The shared HTTP client, its policy and the run tokens.

    Returns:
        The endpoint's reranker; an unknown API format is refused here.

    Example:
        >>> reranker = await open_reranker_at(endpoint, shared)
    """
    http = shared.http
    return open_api_reranker(
        await http.client(), endpoint, tokens=http.run_tokens, policy=http.policy
    )
