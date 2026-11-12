"""OpenAI's chat completions API format: messages in, ``choices[0]`` out.

OpenAI, vLLM, LiteLLM, Ollama's OpenAI route and the Domino AI Gateway all
speak it.

Example:
    >>> OpenAIChatFormat().answer({"choices": [{"message": {"content": "Paris."}}]})
    'Paris.'
"""

from __future__ import annotations

from ...domain.standard_formats import OpenAIChatRequest, OpenAIChatResponse
from .api_format_bases import BaseGenerationApiFormat


class OpenAIChatFormat(BaseGenerationApiFormat):
    """The chat shape of OpenAI's ``POST /v1/chat/completions``.

    Example:
        >>> OpenAIChatFormat().request_body("gpt-4o-mini", messages, 0.0)
        {'model': 'gpt-4o-mini', 'messages': [...], 'temperature': 0.0}
    """

    def request_body(
        self, model: str, messages: list[dict[str, str]], temperature: float
    ) -> dict[str, object]:
        """Build ``{model, messages, temperature}``.

        Args:
            model: The model name sent with the request.
            messages: OpenAI-style messages, each with ``role`` and ``content``.
            temperature: The sampling temperature.

        Returns:
            The JSON body.
        """
        request = OpenAIChatRequest(
            model=model, messages=messages, temperature=temperature
        )
        return request.to_wire()

    def answer(self, body: object) -> str:
        """Read ``choices[0].message.content``.

        Args:
            body: The decoded JSON answer.

        Returns:
            The answer text; "" when the content is null.

        Raises:
            pydantic.ValidationError: If ``body`` is not a chat answer, or has no
                choice.
        """
        response = OpenAIChatResponse.model_validate(body)
        return response.choices[0].message.content or ""
