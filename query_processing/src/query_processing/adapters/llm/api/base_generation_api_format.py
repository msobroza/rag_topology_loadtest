"""What a generation API format is, and the registry of them by name.

An API format is a request body builder and an answer reader, no HTTP: the
shapes are the standard formats of ``domain/standard_formats.py``.

A new format is one new module beside this one, its class decorated with
``@GENERATION_API_FORMATS.register(name)``; the package imports every module.

Example:
    >>> GENERATION_API_FORMATS.get("openai")
    <class 'OpenAIChatFormat'>
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.registry import Registry


class BaseGenerationApiFormat(ABC):
    """A chat wire shape: messages in, the answer text out.

    Example:
        >>> OpenAIChatFormat().answer({"choices": [{"message": {"content": "Hi"}}]})
        'Hi'
    """

    @abstractmethod
    def request_body(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Build the body that asks ``model`` to answer ``messages``.

        Args:
            model: The model name sent with the request.
            messages: OpenAI-style messages, each with ``role`` and ``content``.
            temperature: The sampling temperature.
            response_format: OpenAI's ``response_format`` object, which states
                the schema the answer must follow; None sends none.

        Returns:
            The JSON body.
        """

    @abstractmethod
    def answer(self, body: object) -> str:
        """Read the answer text out of a decoded answer.

        Args:
            body: The decoded JSON answer.

        Returns:
            The answer text; "" when the answer carries no content.

        Raises:
            pydantic.ValidationError: If ``body`` is not this format's answer.
        """


# Every generation format by name; an LLM's api.api_format names one.
GENERATION_API_FORMATS: Registry[type[BaseGenerationApiFormat]] = Registry(
    "generation API format"
)
