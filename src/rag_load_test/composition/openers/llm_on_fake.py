"""The LLM on the ``fake`` backend: an echo, no call out.

Example:
    >>> llm = await open_llm_on_fake(llm_settings, shared)
"""

from __future__ import annotations

from ...adapters.llm.base_llm import BaseLLM
from ...adapters.llm.fake.fake_llm import FakeLLM
from ...config.component_settings import LlmSettings
from .component_openers import LLM_OPENERS, SharedResources


@LLM_OPENERS.register("fake")
async def open_llm_on_fake(
    _component: LlmSettings, _shared: SharedResources
) -> BaseLLM:
    """Open the fake LLM, which takes no settings and shares nothing.

    Args:
        _component: Unused: the fake has no block.
        _shared: Unused.

    Returns:
        A ``FakeLLM``.

    Example:
        >>> await open_llm_on_fake(llm_settings, shared)
    """
    return FakeLLM()
