"""The reranker on the ``none`` backend: retrieval order kept, nothing rescored.

Example:
    >>> reranker = await open_reranker_on_none(reranker_settings, shared)
"""

from __future__ import annotations

from ...adapters.reranker.base_reranker import BaseReranker
from ...adapters.reranker.none.no_reranker import NoReranker
from ...config.components.component_settings import RerankerSettings
from .component_openers import RERANKER_OPENERS, SharedResources


@RERANKER_OPENERS.register("none")
async def open_reranker_on_none(
    _component: RerankerSettings, _shared: SharedResources
) -> BaseReranker:
    """Open the reranker that keeps the retrieval order.

    Args:
        _component: Unused: ``none`` has no block.
        _shared: Unused.

    Returns:
        A ``NoReranker``.

    Example:
        >>> await open_reranker_on_none(reranker_settings, shared)
    """
    return NoReranker()
