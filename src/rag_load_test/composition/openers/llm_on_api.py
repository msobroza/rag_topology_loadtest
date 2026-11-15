"""The LLM on the ``api`` backend: an endpoint in a generation API format.

Example:
    >>> llm = await open_llm_on_api(llm_settings, shared)
"""

from __future__ import annotations

import dataclasses

from ...adapters.llm.api.api_llm import open_api_llm
from ...adapters.llm.base_llm import BaseLLM
from ...config.component_settings import LlmSettings
from .component_openers import LLM_OPENERS, SharedResources


@LLM_OPENERS.register("api")
async def open_llm_on_api(component: LlmSettings, shared: SharedResources) -> BaseLLM:
    """Open the LLM of the endpoint the ``api`` block names, under its own timeout.

    Args:
        component: The LLM's ``api`` block and the timeout of its calls.
        shared: The shared HTTP client, its retries and backoff, and the run
            tokens.

    Returns:
        The endpoint's LLM; an unknown API format is refused here.

    Example:
        >>> llm = await open_llm_on_api(llm_settings, shared)
    """
    http, api = shared.http, component.api
    # An answer takes longer than a model-server call: its own timeout, the
    # shared retries and backoff.
    policy = dataclasses.replace(http.policy, timeout_s=component.http.timeout_s)
    return open_api_llm(
        await http.client(),
        api.endpoint(component.alias),
        tokens=http.run_tokens,
        policy=policy,
        temperature=api.temperature,
    )
