"""The reranker on the ``api`` backend: an endpoint in a rerank API format.

Example:
    >>> reranker = await open_reranker_on_api(components.reranker, shared)
"""

from __future__ import annotations

from ...adapters.reranker.api.api_reranker import open_api_reranker
from ...adapters.reranker.base_reranker import BaseReranker
from ...config.component_settings import RerankerSettings
from .component_openers import RERANKER_OPENERS, SharedResources


@RERANKER_OPENERS.register("api")
async def open_reranker_on_api(
    component: RerankerSettings, shared: SharedResources
) -> BaseReranker:
    """Open the reranker of the endpoint the ``api`` block names.

    Args:
        component: The reranker's ``api`` block.
        shared: The shared HTTP client, its policy and the run tokens.

    Returns:
        The endpoint's reranker; an unknown API format is refused here.

    Example:
        >>> reranker = await open_reranker_on_api(components.reranker, shared)
    """
    http = shared.http
    return open_api_reranker(
        await http.client(),
        component.api.endpoint(),
        tokens=http.run_tokens,
        policy=http.policy,
    )
