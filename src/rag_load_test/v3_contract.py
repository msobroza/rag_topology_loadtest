"""The /v3 contract: the paths and JSON shapes a model server speaks.

OpenVINO Model Server defines it; the ``ovms`` adapters (client side) and the
FastAPI model server (server side) both read it from here, so the two cannot
drift apart.

Example::

    request = RerankRequest(model="bge-reranker-base", query="q", documents=["a"])
    httpx.post(base_url + RERANK_PATH, json=request.to_wire())
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from .models import ScoredPassage, top_by_score

EMBEDDINGS_PATH = "/v3/embeddings"
RERANK_PATH = "/v3/rerank"
MODEL_PROBE_ROUTE = "/v3/models/{name}"
# Verbatim from OVMS: the body of its 404 when ``model`` names no served model.
UNKNOWN_MODEL_ERROR = "Mediapipe graph definition with requested name is not found"


def model_probe_path(alias: str) -> str:
    """The readiness probe path for the served model named ``alias``.

    Example::

        model_probe_path("bge-reranker-base")  # "/v3/models/bge-reranker-base"
    """
    return MODEL_PROBE_ROUTE.format(name=alias)


# --- requests --------------------------------------------------------------


class V3Request(BaseModel):
    """A request body; ``to_wire`` is what goes on the wire."""

    def to_wire(self) -> dict[str, object]:
        """The JSON body, with unset optional fields left out, never null.

        Example::

            RerankRequest(model="r", query="q", documents=[]).to_wire()  # no top_n
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
    """The rerank request for ``passages``; result ``index`` i names ``passages[i]``.

    Example::

        rerank_request_for("bge-reranker-base", "vacation", passages, top_n=5)
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
    """The vectors in request ``input`` order (rows re-ordered by ``index``).

    Example::

        vectors_from_embeddings(EmbeddingsResponse.model_validate(body))
    """
    rows = sorted(response.data, key=lambda row: row.index)
    return [row.embedding for row in rows]


def passages_from_rerank(
    response: RerankResponse, passages: Sequence[ScoredPassage], top_n: int
) -> list[ScoredPassage]:
    """The ``top_n`` best of ``passages``, stamped with their rerank scores.

    ``passages`` are the ones the request was built from; each result names one
    by ``index``. An index outside them raises ``IndexError`` (malformed answer).

    Example::

        passages_from_rerank(RerankResponse.model_validate(body), passages, 5)
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
    """The request's texts as a list (``input`` may be a single string).

    Example::

        embeddings_texts(EmbeddingsRequest(model="e", input="hi"))  # ["hi"]
    """
    return [request.input] if isinstance(request.input, str) else request.input


def embeddings_response_for(
    alias: str, vectors: Sequence[Sequence[float]]
) -> EmbeddingsResponse:
    """The embeddings answer: row ``index`` i holds the vector of input text i.

    Example::

        embeddings_response_for("bge-small-en-v1.5", [[0.1, 0.2]])
    """
    rows = [
        EmbeddingRow(index=i, embedding=list(vector))
        for i, vector in enumerate(vectors)
    ]
    return EmbeddingsResponse(model=alias, data=rows)


def candidates_from_rerank(request: RerankRequest) -> list[ScoredPassage]:
    """The documents as passages whose ``id`` is their index in the request.

    Example::

        candidates_from_rerank(request)[1].id  # "1"
    """
    return [
        ScoredPassage(id=str(i), text=text, retrieval_score=0.0)
        for i, text in enumerate(request.documents)
    ]


def rerank_response_for(ranked: Sequence[ScoredPassage]) -> RerankResponse:
    """The rerank answer for passages made by ``candidates_from_rerank``, best first.

    Example::

        rerank_response_for(await reranker.rerank(query, candidates, top_n))
    """
    return RerankResponse(results=[_rerank_result(passage) for passage in ranked])


def _rerank_result(passage: ScoredPassage) -> RerankResult:
    if passage.rerank_score is None:
        raise ValueError(
            f"passage {passage.id!r} has no rerank_score; expected a reranked "
            "passage from candidates_from_rerank"
        )
    return RerankResult(index=int(passage.id), relevance_score=passage.rerank_score)
