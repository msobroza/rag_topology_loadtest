"""What an embeddings API format is.

An API format is a request body builder and an answer reader, no HTTP: the
shapes are the standard formats of ``domain/standard_formats.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseEmbeddingApiFormat(ABC):
    """An embeddings wire shape: texts in, one vector per text out.

    Example:
        >>> OpenAIEmbeddingsFormat().request_body("bge-small", ["a"])
        {'model': 'bge-small', 'input': ['a']}
    """

    @abstractmethod
    def request_body(self, model: str, texts: Sequence[str]) -> dict[str, object]:
        """Build the body that asks ``model`` to embed ``texts``.

        Args:
            model: The model name sent with the request.
            texts: The texts to embed, in order.

        Returns:
            The JSON body.
        """

    @abstractmethod
    def vectors(self, body: object) -> list[list[float]]:
        """Read the vectors out of a decoded answer.

        Args:
            body: The decoded JSON answer.

        Returns:
            One vector per text sent, in request order.

        Raises:
            pydantic.ValidationError: If ``body`` is not this format's answer.
        """
