"""What an embeddings API format is, and the registry of them by name.

An API format is a request body builder and an answer reader, no HTTP: the
shapes are the standard formats of ``domain/standard_formats.py``. A format is
for an API that does the same operation with other field names: one request
in, all the vectors out. An API that works differently is not a format: it is
an implementation of the port.

A new format is one new module beside this one, its class decorated with
``@EMBEDDING_API_FORMATS.register(name)``; the package imports every module.

Example:
    >>> EMBEDDING_API_FORMATS.get("openai")
    <class 'OpenAIEmbeddingsFormat'>
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ....domain.registry import Registry


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


# Every embeddings format by name; an embedder's api.api_format names one.
EMBEDDING_API_FORMATS: Registry[type[BaseEmbeddingApiFormat]] = Registry(
    "embedding API format"
)
