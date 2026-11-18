"""What a generation API format is, and the registry of them by name.

An API format is a request body builder and an answer reader, no HTTP: the
shapes are the standard formats of ``domain/endpoints/standard_formats.py``.

A new format is one new module beside this one, its class decorated with
``@GENERATION_API_FORMATS.register(name)``; the package imports every module.

Example:
    >>> GENERATION_API_FORMATS.get("openai")
    <class 'OpenAIChatFormat'>
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.components.component_requests import GenerateAnswer
from ....domain.registry import Registry


class BaseGenerationApiFormat(ABC):
    """A chat wire shape: messages in, the answer and its token counts out.

    Example:
        >>> chat = OpenAIChatFormat()
        >>> chat.answer({"choices": [{"message": {"content": "Hi"}}]}).text
        'Hi'
    """

    @abstractmethod
    def request_body(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        *,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, object]:
        """Build the body that asks ``model`` to answer ``messages``.

        The two optional parameters are keyword-only, so a call cannot put a
        cap where the response format goes.

        Args:
            model: The model name sent with the request.
            messages: OpenAI-style messages, each with ``role`` and ``content``.
            temperature: The sampling temperature.
            response_format: OpenAI's ``response_format`` object, which states
                the schema the answer must follow; None sends none.
            max_tokens: The most tokens the answer may hold; None sends no cap.

        Returns:
            The JSON body.
        """

    @abstractmethod
    def answer(self, body: object) -> GenerateAnswer:
        """Read the answer and its token counts out of a decoded answer.

        Args:
            body: The decoded JSON answer.

        Returns:
            The answer text ("" when the answer carries no content) and the
            token counts the answer reports (None for each one it does not).

        Raises:
            pydantic.ValidationError: If ``body`` is not this format's answer.
        """


# Every generation format by name; an LLM's api.api_format names one.
GENERATION_API_FORMATS: Registry[type[BaseGenerationApiFormat]] = Registry(
    "generation API format"
)
