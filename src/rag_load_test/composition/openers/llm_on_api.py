"""The LLM on the ``api`` backend: an endpoint in a generation API format.

``open_llm_at`` opens the LLM of any endpoint: the ``served`` backend uses it
too, with the endpoint it resolves its own way.

Example:
    >>> llm = await open_llm_on_api(llm_settings, shared)
"""

from __future__ import annotations

import dataclasses

from ...adapters.llm.api.api_llm import open_api_llm
from ...adapters.llm.base_llm import BaseLLM
from ...config.component_settings import LlmServedSettings, LlmSettings
from ...domain.api_endpoint import ApiEndpoint
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
    api = component.api
    return await open_llm_at(api.endpoint(component.alias), api, component, shared)


async def open_llm_at(
    endpoint: ApiEndpoint,
    block: LlmServedSettings,
    component: LlmSettings,
    shared: SharedResources,
) -> BaseLLM:
    """Open the LLM of one endpoint, however it was resolved, under its own timeout.

    Args:
        endpoint: The route, its API format, its key and the model name sent.
        block: The block of the component's backend, for the temperature sent.
        component: The LLM, for the timeout of its calls.
        shared: The shared HTTP client, its retries and backoff, and the run
            tokens.

    Returns:
        The endpoint's LLM; an unknown API format is refused here.

    Example:
        >>> llm = await open_llm_at(endpoint, component.api, component, shared)
    """
    http = shared.http
    # An answer takes longer than a model-server call: its own timeout, the
    # shared retries and backoff.
    policy = dataclasses.replace(http.policy, timeout_s=component.http.timeout_s)
    return open_api_llm(
        await http.client(),
        endpoint,
        tokens=http.run_tokens,
        policy=policy,
        temperature=block.temperature,
    )
