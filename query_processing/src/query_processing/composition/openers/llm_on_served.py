"""The LLM on the ``served`` backend: a model server of ours.

It is called as an ``api`` LLM is, at the endpoint ``served_endpoint``
resolves: the route of the ``served`` block, the standard chat format, the
alias, and the run token on Domino. Nothing of ours serves an LLM yet; the
backend is there so that a workflow app can call one when something does.

Example:
    >>> llm = await open_llm_on_served(llm_settings, shared)
"""

from __future__ import annotations

from ...adapters.llm.base_llm import BaseLLM
from ...config.component_settings import LlmSettings
from .component_openers import LLM_OPENERS, SharedResources
from .llm_on_api import open_llm_at


@LLM_OPENERS.register("served")
async def open_llm_on_served(
    component: LlmSettings, shared: SharedResources
) -> BaseLLM:
    """Open the LLM of the model server the ``served`` block names.

    Args:
        component: The LLM's alias, ``served`` and ``http`` blocks.
        shared: The shared HTTP client, the default policy, the run tokens and
            where the process runs.

    Returns:
        The model server's LLM.

    Example:
        >>> llm = await open_llm_on_served(llm_settings, shared)
    """
    endpoint = shared.http.served_endpoint(component)
    return await open_llm_at(endpoint, component.served, component, shared)
