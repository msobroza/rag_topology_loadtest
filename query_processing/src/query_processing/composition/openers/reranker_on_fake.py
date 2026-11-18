"""The reranker on the ``fake`` backend: token overlap with the query, no model.

Example:
    >>> reranker = await open_reranker_on_fake(reranker_settings, shared)
"""

from __future__ import annotations

from ...adapters.reranker.base_reranker import BaseReranker
from ...adapters.reranker.fake.fake_reranker import FakeReranker
from ...config.components.component_settings import RerankerSettings
from .component_openers import RERANKER_OPENERS, SharedResources


@RERANKER_OPENERS.register("fake")
async def open_reranker_on_fake(
    _component: RerankerSettings, _shared: SharedResources
) -> BaseReranker:
    """Open the fake reranker, which takes no settings and shares nothing.

    Args:
        _component: Unused: the fake has no block.
        _shared: Unused.

    Returns:
        A ``FakeReranker``.

    Example:
        >>> await open_reranker_on_fake(reranker_settings, shared)
    """
    return FakeReranker()
