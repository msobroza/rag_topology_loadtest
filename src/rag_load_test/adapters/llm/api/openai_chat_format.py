"""OpenAI's chat completions API format: messages in, ``choices[0]`` out.

OpenAI, vLLM, LiteLLM, Ollama's OpenAI route and the Domino AI Gateway all
speak it.

Example:
    >>> chat = OpenAIChatFormat()
    >>> chat.generation({"choices": [{"message": {"content": "Paris."}}]}).text
    'Paris.'
"""

from __future__ import annotations

from ....domain.generation import Generation
from ....domain.standard_formats import OpenAIChatRequest, OpenAIChatResponse
from .base_generation_api_format import BaseGenerationApiFormat


class OpenAIChatFormat(BaseGenerationApiFormat):
    """The chat shape of OpenAI's ``POST /v1/chat/completions``.

    Example:
        >>> OpenAIChatFormat().request_body("gpt-4o-mini", messages, 0.0, 0)
        {'model': 'gpt-4o-mini', 'messages': [...], 'temperature': 0.0}
    """

    def request_body(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, object]:
        """Build ``{model, messages, temperature}``, and ``max_tokens`` when capped.

        Args:
            model: The model name sent with the request.
            messages: OpenAI-style messages, each with ``role`` and ``content``.
            temperature: The sampling temperature.
            max_tokens: The cap on the answer's tokens; 0 leaves the key out, so
                an uncapped body is the one sent before the cap existed.

        Returns:
            The JSON body.
        """
        request = OpenAIChatRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or None,
        )
        return request.to_wire()

    def generation(self, body: object) -> Generation:
        """Read ``choices[0].message.content`` and the ``usage`` token counts.

        Args:
            body: The decoded JSON answer.

        Returns:
            The answer text ("" when the content is null) and its token counts
            (None for each one ``usage`` lacks, both when there is no ``usage``).

        Raises:
            pydantic.ValidationError: If ``body`` is not a chat answer, or has no
                choice.
        """
        response = OpenAIChatResponse.model_validate(body)
        text = response.choices[0].message.content or ""
        usage = response.usage
        if usage is None:
            return Generation(text)
        return Generation(text, usage.prompt_tokens, usage.completion_tokens)
