"""The reranker on the ``served`` backend: a model server of ours.

It is called as an ``api`` reranker is, at the endpoint ``served_endpoint``
resolves: the route of the ``served`` block, the standard rerank format, the
alias, and the run token on Domino.

Example:
    >>> reranker = await open_reranker_on_served(reranker_settings, shared)
"""

from __future__ import annotations

from ...adapters.reranker.base_reranker import BaseReranker
from ...config.components.component_settings import RerankerSettings
from .component_openers import RERANKER_OPENERS, SharedResources
from .reranker_on_api import open_reranker_at


@RERANKER_OPENERS.register("served")
async def open_reranker_on_served(
    component: RerankerSettings, shared: SharedResources
) -> BaseReranker:
    """Open the reranker of the model server the ``served`` block names.

    Args:
        component: The reranker's alias, ``served`` and ``http`` blocks.
        shared: The shared HTTP client, the default policy, the run tokens and
            where the process runs.

    Returns:
        The model server's reranker.

    Example:
        >>> reranker = await open_reranker_on_served(reranker_settings, shared)
    """
    endpoint = shared.http.served_endpoint(component)
    return await open_reranker_at(endpoint, component, shared)
