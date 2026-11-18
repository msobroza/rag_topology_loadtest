"""OpenAI's chat completions API format: messages in, ``choices[0]`` out.

OpenAI, vLLM, LiteLLM, Ollama's OpenAI route and the Domino AI Gateway all
speak it.

Example:
    >>> OpenAIChatFormat().answer({"choices": [{"message": {"content": "Paris."}}]})
    GenerateAnswer(text='Paris.', prompt_tokens=None, completion_tokens=None)
"""

from __future__ import annotations

from ....domain.components.component_requests import GenerateAnswer
from ....domain.endpoints.standard_formats import (
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChatUsage,
)
from .base_generation_api_format import GENERATION_API_FORMATS, BaseGenerationApiFormat


@GENERATION_API_FORMATS.register("openai")
class OpenAIChatFormat(BaseGenerationApiFormat):
    """The chat shape of OpenAI's ``POST /v1/chat/completions``.

    Example:
        >>> OpenAIChatFormat().request_body("gpt-4o-mini", messages, 0.0)
        {'model': 'gpt-4o-mini', 'messages': [...], 'temperature': 0.0}
    """

    def request_body(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        """Build ``{model, messages, temperature}``, and each optional key set.

        Args:
            model: The model name sent with the request.
            messages: OpenAI-style messages, each with ``role`` and ``content``.
            temperature: The sampling temperature.
            response_format: OpenAI's ``response_format`` object; None leaves
                the key out.
            max_tokens: The cap on the answer's tokens; None leaves the key
                out, so an uncapped body is the one sent before the cap existed.

        Returns:
            The JSON body.
        """
        request = OpenAIChatRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            max_tokens=max_tokens,
        )
        return request.to_wire()

    def answer(self, body: object) -> GenerateAnswer:
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
        usage = response.usage or OpenAIChatUsage()  # not every server reports one
        return GenerateAnswer(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
