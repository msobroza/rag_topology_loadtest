"""What an API format is: a request body builder and an answer reader, no HTTP.

Each format turns the call's arguments into a JSON body and reads the decoded
answer back; the adapters (``ApiEmbedder``, ``ApiReranker``, ``ApiLLM``)
send it. So a new wire shape is one new subclass, and no adapter changes.

Example:
    >>> api_format = OpenAIEmbeddingsFormat()
    >>> body = api_format.request_body("bge-small", ["vacation policy"])
    >>> vectors = api_format.vectors(await endpoint_client.post(body))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel


class BaseApiRequest(BaseModel):
    """A request body of some API format; ``to_wire`` is what goes on the wire.

    Example:
        >>> CohereRerankRequest(model="r", query="q", documents=[]).to_wire()
        {'model': 'r', 'query': 'q', 'documents': []}
    """

    def to_wire(self) -> dict[str, object]:
        """Return the JSON body, with unset optional fields left out, never null.

        Returns:
            The body as a dict, ready for ``json=``.

        Example:
            >>> CohereRerankRequest(model="r", query="q", documents=[]).to_wire()
            {'model': 'r', 'query': 'q', 'documents': []}
        """
        return self.model_dump(exclude_none=True)


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


class BaseRerankApiFormat(ABC):
    """A rerank wire shape: a query and documents in, scores by index out.

    Example:
        >>> body = {"results": [{"index": 1, "relevance_score": 0.9}]}
        >>> CohereRerankFormat().scores(body)
        {1: 0.9}
    """

    @abstractmethod
    def request_body(
        self, model: str, query: str, documents: Sequence[str], top_n: int
    ) -> dict[str, object]:
        """Build the body that asks ``model`` to score ``documents`` against ``query``.

        Args:
            model: The model name sent with the request.
            query: The question the documents are scored against.
            documents: The texts to score; an answer's index i names documents[i].
            top_n: How many results the server keeps.

        Returns:
            The JSON body.
        """

    @abstractmethod
    def scores(self, body: object) -> dict[int, float]:
        """Read the relevance scores out of a decoded answer.

        Args:
            body: The decoded JSON answer.

        Returns:
            Document index to relevance score, in the answer's order. The
            indexes are not checked against the documents sent.

        Raises:
            pydantic.ValidationError: If ``body`` is not this format's answer.
        """


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
