"""What a rerank API format is.

An API format is a request body builder and an answer reader, no HTTP: the
shapes are the standard formats of ``domain/standard_formats.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


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
