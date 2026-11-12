"""OpenAI's embeddings API format: ``{model, input}`` in, ``data[].embedding`` out.

OVMS ``/v3/embeddings``, vLLM, LiteLLM and OpenAI itself all speak it. Rows may
come back in any ``index`` order; they are put back in request order.

Example:
    >>> api_format = OpenAIEmbeddingsFormat()
    >>> api_format.request_body("bge-small", ["a"])
    {'model': 'bge-small', 'input': ['a']}
"""

from __future__ import annotations

from collections.abc import Sequence

from ....domain.standard_formats import (
    OpenAIEmbeddingsRequest,
    OpenAIEmbeddingsResponse,
)
from .base_embedding_api_format import BaseEmbeddingApiFormat


class OpenAIEmbeddingsFormat(BaseEmbeddingApiFormat):
    """The embeddings shape of OpenAI's ``POST /v1/embeddings``.

    Example:
        >>> body = {"data": [{"index": 0, "embedding": [1.0]}]}
        >>> OpenAIEmbeddingsFormat().vectors(body)
        [[1.0]]
    """

    def request_body(self, model: str, texts: Sequence[str]) -> dict[str, object]:
        """Build ``{model, input}`` with ``input`` a list, even for one text.

        Args:
            model: The model name sent with the request.
            texts: The texts to embed, in order.

        Returns:
            The JSON body.
        """
        return OpenAIEmbeddingsRequest(model=model, input=list(texts)).to_wire()

    def vectors(self, body: object) -> list[list[float]]:
        """Read ``data[].embedding``, re-ordered by ``index``.

        Args:
            body: The decoded JSON answer.

        Returns:
            One vector per row, in request order.

        Raises:
            pydantic.ValidationError: If ``body`` is not an embeddings answer, or
                its row indexes are not 0..n-1 once each.
        """
        response = OpenAIEmbeddingsResponse.model_validate(body)
        rows = sorted(response.data, key=lambda row: row.index)
        return [row.embedding for row in rows]
