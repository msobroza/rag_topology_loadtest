"""Cohere's rerank API format: ``{model, query, documents, top_n}`` in, scores out.

Cohere's ``/v2/rerank``, OVMS ``/v3/rerank`` and the ``/rerank`` routes of
LiteLLM, vLLM, Jina and Infinity all speak it.

Example:
    >>> CohereRerankFormat().request_body("rerank-v3.5", "q", ["a", "b"], top_n=1)
    {'model': 'rerank-v3.5', 'query': 'q', 'documents': ['a', 'b'], 'top_n': 1}
"""

from __future__ import annotations

from collections.abc import Sequence

from ....domain.standard_formats import CohereRerankRequest, CohereRerankResponse
from .base_rerank_api_format import BaseRerankApiFormat


class CohereRerankFormat(BaseRerankApiFormat):
    """The rerank shape of Cohere's ``POST /v2/rerank``.

    Example:
        >>> body = {"results": [{"index": 1, "relevance_score": 0.9}]}
        >>> CohereRerankFormat().scores(body)
        {1: 0.9}
    """

    def request_body(
        self, model: str, query: str, documents: Sequence[str], top_n: int
    ) -> dict[str, object]:
        """Build ``{model, query, documents, top_n}``.

        Args:
            model: The model name sent with the request.
            query: The question the documents are scored against.
            documents: The texts to score, in order.
            top_n: How many results the server keeps.

        Returns:
            The JSON body.
        """
        request = CohereRerankRequest(
            model=model, query=query, documents=list(documents), top_n=top_n
        )
        return request.to_wire()

    def scores(self, body: object) -> dict[int, float]:
        """Read ``results[]`` as document index to ``relevance_score``.

        Args:
            body: The decoded JSON answer.

        Returns:
            Document index to relevance score, in the answer's order.

        Raises:
            pydantic.ValidationError: If ``body`` is not a rerank answer.
        """
        response = CohereRerankResponse.model_validate(body)
        return {result.index: result.relevance_score for result in response.results}
