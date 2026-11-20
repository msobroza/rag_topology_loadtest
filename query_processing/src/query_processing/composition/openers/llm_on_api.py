"""The LLM on the ``api`` backend: an endpoint in a generation API format.

``open_llm_at`` opens the LLM of any endpoint: the ``served`` backend uses it
too, with the endpoint it resolves its own way.

Example:
    >>> llm = await open_llm_on_api(llm_settings, shared)
"""

from __future__ import annotations

from ...adapters.key_source.base_key_source import BaseKeySource
from ...adapters.llm.api.api_llm import open_api_llm
from ...adapters.llm.base_llm import BaseLLM
from ...config.components.component_settings import LlmServedSettings, LlmSettings
from ...domain.endpoints.api_endpoint import ApiEndpoint
from .component_openers import LLM_OPENERS, SharedResources


@LLM_OPENERS.register("api")
async def open_llm_on_api(component: LlmSettings, shared: SharedResources) -> BaseLLM:
    """Open the LLM of the endpoint the ``api`` block names.

    Args:
        component: The LLM's alias, ``api`` and ``http`` blocks.
        shared: The shared HTTP client, the default policy and the key sources.

    Returns:
        The endpoint's LLM; an unknown API format is refused here.

    Example:
        >>> llm = await open_llm_on_api(llm_settings, shared)
    """
    api = component.api
    key_source = await shared.http.key_source_of(api.api_key)
    endpoint = api.endpoint(component.alias)
    return await open_llm_at(endpoint, key_source, api, component, shared)


async def open_llm_at(
    endpoint: ApiEndpoint,
    key_source: BaseKeySource | None,
    block: LlmServedSettings,
    component: LlmSettings,
    shared: SharedResources,
) -> BaseLLM:
    """Open the LLM of one endpoint, however it was resolved.

    Args:
        endpoint: The route, its API format and the model name sent.
        key_source: Where its API key comes from; None sends no key.
        block: The block of the component's backend, for the temperature sent
            and the structured output method.
        component: The LLM, for how patiently it is called.
        shared: The shared HTTP client, the default policy and the key sources.

    Returns:
        The endpoint's LLM; an unknown API format is refused here.

    Example:
        >>> llm = await open_llm_at(endpoint, None, component.api, component,
        ...                         shared)
    """
    http = shared.http
    return open_api_llm(
        await http.client(),
        endpoint,
        key_source=key_source,
        policy=http.policy_of(component),
        temperature=block.temperature,
        structured_output=block.structured_output,
    )
