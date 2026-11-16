"""BaseLLM over an HTTP endpoint that speaks a generation API format.

Example:
    >>> llm = ApiLLM(endpoint_client, generation_api_format("openai"),
    ...              temperature=0.0, max_tokens=0)
    >>> generation = await llm.generate(messages)
"""

from __future__ import annotations

import httpx
from pydantic import ValidationError

from ....domain.api_endpoint import ApiEndpoint
from ....domain.generation import Generation
from ....infrastructure.endpoint.api_endpoint_client import ApiEndpointClient
from ....infrastructure.endpoint.open_endpoint_client import (
    RunTokenSource,
    open_endpoint_client,
)
from ....infrastructure.http.http_policy import HttpPolicy
from ..base_llm import BaseLLM
from .base_generation_api_format import BaseGenerationApiFormat
from .generation_api_formats import generation_api_format


class ApiLLM(BaseLLM):
    """Answers conversations on one endpoint at a fixed temperature and cap.

    Args:
        endpoint_client: The endpoint every request goes to (component ``llm``).
        api_format: Builds each request body and reads each answer.
        temperature: The sampling temperature sent with every request.
        max_tokens: The cap on every answer's tokens; 0 sends no cap.

    Example:
        >>> llm = ApiLLM(endpoint_client, OpenAIChatFormat(),
        ...              temperature=0.0, max_tokens=0)
        >>> (await llm.generate([{"role": "user", "content": "Hi"}])).text
        'Hello!'
    """

    def __init__(
        self,
        endpoint_client: ApiEndpointClient,
        api_format: BaseGenerationApiFormat,
        *,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._endpoint_client, self._api_format = endpoint_client, api_format
        self._temperature, self._max_tokens = temperature, max_tokens
        self.model_name = endpoint_client.endpoint.model

    async def generate(self, messages: list[dict[str, str]]) -> Generation:
        """Answer a conversation at the configured temperature and cap.

        Args:
            messages: OpenAI-style messages, each with ``role`` and ``content``.

        Returns:
            The model's answer text ("" when the answer carries no content), with
            the token counts the endpoint reported.

        Raises:
            RagDependencyError: If the request fails (a timeout has kind
                ``timeout``) or the answer is not the format's shape.
        """
        body = self._api_format.request_body(
            self.model_name, messages, self._temperature, self._max_tokens
        )
        answer = await self._endpoint_client.post(body)
        try:
            return self._api_format.generation(answer)
        except ValidationError as exc:
            raise self._endpoint_client.malformed(exc) from exc


def open_api_llm(
    client: httpx.AsyncClient,
    endpoint: ApiEndpoint,
    *,
    tokens: RunTokenSource,
    policy: HttpPolicy,
    temperature: float,
    max_tokens: int,
) -> ApiLLM:
    """Build the LLM of one generation endpoint.

    Args:
        client: The shared HTTP client.
        endpoint: The generation route, its API format, its key and its model.
        tokens: Where the Domino run token comes from, for the ``domino`` key.
        policy: Timeout, retries and backoff of every call; generation has a
            longer timeout than the model servers.
        temperature: The sampling temperature sent with every request.
        max_tokens: The cap on every answer's tokens; 0 sends no cap.

    Returns:
        The LLM.

    Raises:
        ValueError: If ``endpoint.api_format`` names no generation API format.

    Example:
        >>> open_api_llm(client, endpoint, tokens=tokens, policy=policy,
        ...              temperature=0.0, max_tokens=0)
    """
    api_format = generation_api_format(endpoint.api_format)
    endpoint_client = open_endpoint_client(
        client, endpoint, component="llm", tokens=tokens, policy=policy
    )
    return ApiLLM(
        endpoint_client, api_format, temperature=temperature, max_tokens=max_tokens
    )
