"""What a generation API format is.

An API format is a request body builder and an answer reader, no HTTP: the
shapes are the standard formats of ``domain/standard_formats.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ....domain.generation import Generation


class BaseGenerationApiFormat(ABC):
    """A chat wire shape: messages in, the answer and its token counts out.

    Example:
        >>> chat = OpenAIChatFormat()
        >>> chat.generation({"choices": [{"message": {"content": "Hi"}}]}).text
        'Hi'
    """

    @abstractmethod
    def request_body(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, object]:
        """Build the body that asks ``model`` to answer ``messages``.

        Args:
            model: The model name sent with the request.
            messages: OpenAI-style messages, each with ``role`` and ``content``.
            temperature: The sampling temperature.
            max_tokens: The cap on the answer's tokens; 0 sends no cap.

        Returns:
            The JSON body.
        """

    @abstractmethod
    def generation(self, body: object) -> Generation:
        """Read the answer and its token counts out of a decoded answer.

        Args:
            body: The decoded JSON answer.

        Returns:
            The answer text ("" when the answer carries no content) and the
            token counts the answer reports (None for each one it does not).

        Raises:
            pydantic.ValidationError: If ``body`` is not this format's answer.
        """
