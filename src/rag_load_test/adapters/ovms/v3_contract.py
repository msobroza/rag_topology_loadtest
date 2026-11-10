"""The /v3 contract: the paths and JSON shapes a model server speaks.

OpenVINO Model Server defines it; the ``ovms`` adapters (client side) and the
FastAPI model server (server side) both read it from here, so the two cannot
drift apart.

Example:
    >>> request = RerankRequest(model="bge-reranker-base", query="q", documents=["a"])
    >>> httpx.post(base_url + RERANK_PATH, json=request.to_wire())
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from ...domain.passages import ScoredPassage, top_by_score

EMBEDDINGS_PATH = "/v3/embeddings"
RERANK_PATH = "/v3/rerank"
MODEL_PROBE_ROUTE = "/v3/models/{name}"
# Verbatim from OVMS: the body of its 404 when ``model`` names no served model.
UNKNOWN_MODEL_ERROR = "Mediapipe graph definition with requested name is not found"


def model_probe_path(alias: str) -> str:
    """Return the readiness probe path for a served model.

    Args:
        alias: The name the model is served under.

    Returns:
        The path under the server's base URL.

    Example:
        >>> model_probe_path("bge-reranker-base")
        '/v3/models/bge-reranker-base'
    """
    return MODEL_PROBE_ROUTE.format(name=alias)


# --- requests --------------------------------------------------------------


class V3Request(BaseModel):
    """A request body; ``to_wire`` is what goes on the wire."""

    def to_wire(self) -> dict[str, object]:
        """Return the JSON body, with unset optional fields left out, never null.

        Returns:
            The body as a dict, ready for ``json=``.

        Example:
            >>> RerankRequest(model="r", query="q", documents=[]).to_wire()  # no top_n
            {'model': 'r', 'query': 'q', 'documents': []}
        """
        return self.model_dump(exclude_none=True)


class EmbeddingsRequest(V3Request):
    """``POST /v3/embeddings``: one text or a batch of texts for the alias ``model``."""

    model: str
    input: str | list[str]


class RerankRequest(V3Request):
    """``POST /v3/rerank``: score ``documents`` against ``query``; keep ``top_n``."""

    model: str
    query: str
    documents: list[str]
    top_n: int | None = None


def rerank_request_for(
    alias: str, query: str, passages: Sequence[ScoredPassage], top_n: int
) -> RerankRequest:
    """Build the rerank request for ``passages``.

    Args:
        alias: The name the reranker is served under.
        query: The question the passages are scored against.
        passages: The candidates; result ``index`` i names ``passages[i]``.
        top_n: How many results the server keeps.

    Returns:
        The request, with the passages' texts as ``documents``, in order.

    Example:
        >>> rerank_request_for("bge-reranker-base", "vacation", passages, top_n=5)
    """
    documents = [passage.text for passage in passages]
    return RerankRequest(model=alias, query=query, documents=documents, top_n=top_n)


# --- responses -------------------------------------------------------------
# Readers ignore fields they do not use (OVMS also sends ``usage``); the
# ``object`` and ``model`` defaults are what the FastAPI model server sends.


class EmbeddingRow(BaseModel):
    """One vector; ``index`` is the position of its text in the request ``input``."""

    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    """``POST /v3/embeddings`` answer; rows may come in any ``index`` order."""

    object: str = "list"
    model: str = ""
    data: list[EmbeddingRow]


class RerankResult(BaseModel):
    """One score; ``index`` is the position of its document in the request."""

    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    """``POST /v3/rerank`` answer: at most ``top_n`` results."""

    results: list[RerankResult]


# --- index bookkeeping, client side ----------------------------------------


def vectors_from_embeddings(response: EmbeddingsResponse) -> list[list[float]]:
    """Return the vectors in request ``input`` order.

    Args:
        response: The embeddings answer; its rows may come in any ``index`` order.

    Returns:
        One vector per row, re-ordered by ``index``.

    Example:
        >>> vectors_from_embeddings(EmbeddingsResponse.model_validate(body))
    """
    rows = sorted(response.data, key=lambda row: row.index)
    return [row.embedding for row in rows]


def passages_from_rerank(
    response: RerankResponse, passages: Sequence[ScoredPassage], top_n: int
) -> list[ScoredPassage]:
    """Return the ``top_n`` best of ``passages``, stamped with their rerank scores.

    An index outside ``passages`` raises ``IndexError`` (malformed answer).

    Args:
        response: The rerank answer; each result names one passage by ``index``.
        passages: The ones the request was built from, in the same order.
        top_n: How many passages to keep.

    Returns:
        At most ``top_n`` of the passages the results name, best first.

    Example:
        >>> passages_from_rerank(RerankResponse.model_validate(body), passages, 5)
    """
    scores = {result.index: result.relevance_score for result in response.results}
    chosen = [_passage_at(passages, index) for index in scores]
    return top_by_score(chosen, list(scores.values()), top_n)


def _passage_at(passages: Sequence[ScoredPassage], index: int) -> ScoredPassage:
    # Negative indexes would silently wrap around in Python: reject them too.
    if not 0 <= index < len(passages):
        raise IndexError(
            f"rerank result index {index} is outside the {len(passages)} documents "
            f"sent (expected 0..{len(passages) - 1})"
        )
    return passages[index]


# --- index bookkeeping, server side ----------------------------------------


def embeddings_texts(request: EmbeddingsRequest) -> list[str]:
    """Return the request's texts as a list.

    Args:
        request: The embeddings request; its ``input`` may be a single string.

    Returns:
        The texts, in ``input`` order.

    Example:
        >>> embeddings_texts(EmbeddingsRequest(model="e", input="hi"))
        ['hi']
    """
    return [request.input] if isinstance(request.input, str) else request.input


def embeddings_response_for(
    alias: str, vectors: Sequence[Sequence[float]]
) -> EmbeddingsResponse:
    """Build the embeddings answer.

    Args:
        alias: The name the embedder is served under, sent back as ``model``.
        vectors: One vector per input text, in input order.

    Returns:
        The answer, where row ``index`` i holds the vector of input text i.

    Example:
        >>> embeddings_response_for("bge-small-en-v1.5", [[0.1, 0.2]])
    """
    rows = [
        EmbeddingRow(index=i, embedding=list(vector))
        for i, vector in enumerate(vectors)
    ]
    return EmbeddingsResponse(model=alias, data=rows)


def candidates_from_rerank(request: RerankRequest) -> list[ScoredPassage]:
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


def rerank_response_for(ranked: Sequence[ScoredPassage]) -> RerankResponse:
    """Build the rerank answer.

    Args:
        ranked: Reranked passages made by ``candidates_from_rerank``, best first;
            each must carry a ``rerank_score``.

    Returns:
        One result per passage, in the given order, ``index`` taken from its id.

    Example:
        >>> rerank_response_for(await reranker.rerank(query, candidates, top_n))
    """
    return RerankResponse(results=[_rerank_result(passage) for passage in ranked])


def _rerank_result(passage: ScoredPassage) -> RerankResult:
    if passage.rerank_score is None:
        raise ValueError(
            f"passage {passage.id!r} has no rerank_score; expected a reranked "
            "passage from candidates_from_rerank"
        )
    return RerankResult(index=int(passage.id), relevance_score=passage.rerank_score)
