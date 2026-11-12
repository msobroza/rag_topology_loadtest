"""The server side of our model servers: their answers, in the API formats' shapes.

The FastAPI model server answers on OpenVINO Model Server's routes (the
``MODEL_SERVER_*`` constants of ``domain/model_api.py``) in the same formats the
``api`` adapters speak: OpenAI's embeddings shape and Cohere's rerank shape. It
reads requests with the standards' own models (``OpenAIEmbeddingsRequest``,
``CohereRerankRequest`` of ``domain/standard_formats.py``), so client and server
cannot drift apart.

Example:
    >>> candidates = candidates_from_rerank(request)
    >>> rerank_response_for(await reranker.rerank(request.query, candidates, 5))
"""

from __future__ import annotations

from collections.abc import Sequence

from ...domain.passages import ScoredPassage
from ...domain.standard_formats import (
    CohereRerankRequest,
    CohereRerankResponse,
    CohereRerankResult,
    OpenAIEmbeddingRow,
    OpenAIEmbeddingsResponse,
)

# Verbatim from OVMS: the body of its 404 when ``model`` names no served model.
UNKNOWN_MODEL_ERROR = "Mediapipe graph definition with requested name is not found"


def embeddings_response_for(
    alias: str, vectors: Sequence[Sequence[float]]
) -> OpenAIEmbeddingsResponse:
    """Build the embeddings answer.

    Args:
        alias: The name the embedder is served under, sent back as ``model``.
        vectors: One vector per input text, in input order.

    Returns:
        The answer, where row ``index`` i holds the vector of input text i.

    Example:
        >>> embeddings_response_for("embedder-alias", [[0.1, 0.2]]).data[0].index
        0
    """
    rows = [
        OpenAIEmbeddingRow(index=i, embedding=list(vector))
        for i, vector in enumerate(vectors)
    ]
    return OpenAIEmbeddingsResponse(model=alias, data=rows)


def candidates_from_rerank(request: CohereRerankRequest) -> list[ScoredPassage]:
    """Turn the request's documents into rerank candidates.

    Args:
        request: The rerank request the server received.

    Returns:
        One passage per document, whose ``id`` is its index in the request and
        whose ``retrieval_score`` is 0.0.

    Example:
        >>> candidates_from_rerank(request)[1].id
        '1'
    """
    return [
        ScoredPassage(id=str(i), text=text, retrieval_score=0.0)
        for i, text in enumerate(request.documents)
    ]


def rerank_response_for(ranked: Sequence[ScoredPassage]) -> CohereRerankResponse:
    """Build the rerank answer.

    Args:
        ranked: Reranked passages made by ``candidates_from_rerank``, best first;
            each must carry a ``rerank_score``.

    Returns:
        One result per passage, in the given order, ``index`` taken from its id.

    Raises:
        ValueError: If a passage has no ``rerank_score``.

    Example:
        >>> rerank_response_for(await reranker.rerank(query, candidates, top_n))
    """
    return CohereRerankResponse(results=[_rerank_result(p) for p in ranked])


def _rerank_result(passage: ScoredPassage) -> CohereRerankResult:
    if passage.rerank_score is None:
        raise ValueError(
            f"passage {passage.id!r} has no rerank_score; expected a reranked "
            "passage from candidates_from_rerank"
        )
    return CohereRerankResult(
        index=int(passage.id), relevance_score=passage.rerank_score
    )
