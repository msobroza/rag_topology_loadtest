"""The embedder on the ``fake`` backend: a hashed bag of words, no model.

Example:
    >>> embedder = await open_embedder_on_fake(components.embedder, shared)
"""

from __future__ import annotations

from ...adapters.embedder.base_embedder import BaseEmbedder
from ...adapters.embedder.fake.fake_embedder import FakeEmbedder
from ...config.component_settings import EmbedderSettings
from .component_openers import EMBEDDER_OPENERS, SharedResources


@EMBEDDER_OPENERS.register("fake")
async def open_embedder_on_fake(
    _component: EmbedderSettings, _shared: SharedResources
) -> BaseEmbedder:
    """Open the fake embedder, which takes no settings and shares nothing.

    Args:
        _component: Unused: the fake has no block.
        _shared: Unused.

    Returns:
        A ``FakeEmbedder``.

    Example:
        >>> await open_embedder_on_fake(components.embedder, shared)
    """
    return FakeEmbedder()
