"""The LLM on the ``api`` backend: an endpoint in a generation API format.

Example:
    >>> llm = await open_llm_on_api(components.llm, shared)
"""

from __future__ import annotations

from ...adapters.llm.api.api_llm import open_api_llm
from ...adapters.llm.base_llm import BaseLLM
from ...config.component_settings import LlmSettings
from ...infrastructure.http.http_policy import HttpPolicy
from .component_openers import LLM_OPENERS, SharedResources


@LLM_OPENERS.register("api")
async def open_llm_on_api(component: LlmSettings, shared: SharedResources) -> BaseLLM:
    """Open the LLM of the endpoint the ``api`` block names, under its own policy.

    Args:
        component: The LLM's ``api`` block and the HTTP settings of its calls.
        shared: The shared HTTP client and the run tokens.

    Returns:
        The endpoint's LLM; an unknown API format is refused here.

    Example:
        >>> llm = await open_llm_on_api(components.llm, shared)
    """
    http, api = component.http, component.api
    policy = HttpPolicy(http.timeout_s, http.retries, http.backoff_s)
    return open_api_llm(
        await shared.http.client(),
        api.endpoint(),
        tokens=shared.http.run_tokens,
        policy=policy,
        temperature=api.temperature,
    )
