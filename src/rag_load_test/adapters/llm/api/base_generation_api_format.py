"""What a generation API format is.

An API format is a request body builder and an answer reader, no HTTP: the
shapes are the standard formats of ``domain/standard_formats.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGenerationApiFormat(ABC):
    """A chat wire shape: messages in, the answer text out.

    Example:
        >>> OpenAIChatFormat().answer({"choices": [{"message": {"content": "Hi"}}]})
        'Hi'
    """

    @abstractmethod
    def request_body(
        self, model: str, messages: list[dict[str, str]], temperature: float
    ) -> dict[str, object]:
        """Build the body that asks ``model`` to answer ``messages``.

        Args:
            model: The model name sent with the request.
            messages: OpenAI-style messages, each with ``role`` and ``content``.
            temperature: The sampling temperature.

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
