"""BaseLLM over an HTTP endpoint that speaks a generation API format.

Example:
    >>> chat_model = ApiLLM(endpoint_client, generation_api_format("openai"),
    ...                           temperature=0.0)
    >>> answer = await chat_model.generate(messages)
"""

from __future__ import annotations

from pydantic import ValidationError

from ...domain.ports import BaseLLM
from .api_endpoint_client import ApiEndpointClient
from .api_format_bases import BaseGenerationApiFormat


class ApiLLM(BaseLLM):
    """Answers conversations on one endpoint at a fixed temperature.

    Args:
        endpoint_client: The endpoint every request goes to (component ``llm``).
        api_format: Builds each request body and reads each answer.
        temperature: The sampling temperature sent with every request.

    Example:
        >>> chat_model = ApiLLM(endpoint_client, OpenAIChatFormat(),
        ...                           temperature=0.0)
        >>> await chat_model.generate([{"role": "user", "content": "Hi"}])
        'Hello!'
    """

    def __init__(
        self,
        endpoint_client: ApiEndpointClient,
        api_format: BaseGenerationApiFormat,
        *,
        temperature: float,
    ) -> None:
        self._endpoint_client, self._api_format = endpoint_client, api_format
        self._temperature = temperature
        self.model_name = endpoint_client.endpoint.model

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Answer a conversation at the configured temperature.

        Args:
            messages: OpenAI-style messages, each with ``role`` and ``content``.

        Returns:
            The model's answer text; "" when the answer carries no content.

        Raises:
            RagDependencyError: If the request fails (a timeout has kind
                ``timeout``) or the answer is not the format's shape.
        """
        body = self._api_format.request_body(
            self.model_name, messages, self._temperature
        )
        answer = await self._endpoint_client.post(body)
        try:
            return self._api_format.answer(answer)
        except ValidationError as exc:
            raise self._endpoint_client.malformed(exc) from exc
