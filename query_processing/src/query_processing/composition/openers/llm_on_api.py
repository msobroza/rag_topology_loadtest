"""The LLM on the ``api`` backend: an endpoint in a generation API format.

``open_llm_at`` opens the LLM of any endpoint: the ``served`` backend uses it
too, with the endpoint it resolves its own way.

Example:
    >>> llm = await open_llm_on_api(llm_settings, shared)
"""

from __future__ import annotations

from ...adapters.llm.api.api_llm import open_api_llm
from ...adapters.llm.base_llm import BaseLLM
from ...config.component_settings import LlmServedSettings, LlmSettings
from ...domain.api_endpoint import ApiEndpoint
from .component_openers import LLM_OPENERS, SharedResources


@LLM_OPENERS.register("api")
async def open_llm_on_api(component: LlmSettings, shared: SharedResources) -> BaseLLM:
    """Open the LLM of the endpoint the ``api`` block names.

    Args:
        component: The LLM's alias, ``api`` and ``http`` blocks.
        shared: The shared HTTP client, the default policy and the run tokens.

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
    """Open the LLM of one endpoint, however it was resolved.

    Args:
        endpoint: The route, its API format, its key and the model name sent.
        block: The block of the component's backend, for the temperature sent
            and the structured output method.
        component: The LLM, for how patiently it is called.
        shared: The shared HTTP client, the default policy and the run tokens.

    Returns:
        The endpoint's LLM; an unknown API format is refused here.

    Example:
        >>> llm = await open_llm_at(endpoint, component.api, component, shared)
    """
    http = shared.http
    return open_api_llm(
        await http.client(),
        endpoint,
        tokens=http.run_tokens,
        policy=http.policy_of(component),
        temperature=block.temperature,
        structured_output=block.structured_output,
    )
