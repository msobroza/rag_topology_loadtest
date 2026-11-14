"""A FastAPI model server on OVMS's routes and formats (the FastAPI-vs-OVMS experiment).

Like an OVMS app it serves ONE model per process: ``--serve embedder`` exposes
``POST /v3/embeddings`` (OpenAI's embeddings format), ``--serve reranker``
exposes ``POST /v3/rerank`` (Cohere's rerank format). Both answer
``GET /v3/models/{name}``, as OVMS does. Routes come from ``domain/api_endpoint.py``
and bodies are the standard formats' own models (``OpenAIEmbeddingsRequest``,
``CohereRerankRequest`` of ``domain/standard_formats.py``), the shapes the ``api``
adapters speak, so client and server cannot drift apart; the answers are built
by the three functions at the end of this module. Like OVMS, a request for any
model but this server's alias is a 404. The workflow app cannot
tell the two servers apart, so ``RAG_EMBEDDING_API_URL`` / ``RAG_RERANK_API_URL``
can point at either and the same load test compares FastAPI with OVMS.

Example:
    $ rag-model-server --serve reranker --port 8011
    $ RAG_TOPOLOGY=split-reranker \
    >   RAG_RERANK_API_URL=http://localhost:8011/v3/rerank rag-serve
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import get_args

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from ..adapters.rag_ports import RagPorts
from ..config.rag_settings import RagSettings
from ..domain.api_endpoint import (
    MODEL_SERVER_EMBEDDINGS_PATH,
    MODEL_SERVER_MODEL_ROUTE,
    MODEL_SERVER_RERANK_PATH,
)
from ..domain.passages import ScoredPassage
from ..domain.roles import ModelServerPart
from ..domain.standard_formats import (
    CohereRerankRequest,
    CohereRerankResponse,
    CohereRerankResult,
    OpenAIEmbeddingRow,
    OpenAIEmbeddingsRequest,
    OpenAIEmbeddingsResponse,
)

# Verbatim from OVMS: the body of its 404 when ``model`` names no served model.
UNKNOWN_MODEL_ERROR = "Mediapipe graph definition with requested name is not found"


# The handlers read what the factory left on ``app.state``: ``part``, ``alias``
# and ``ports`` from the start, the served model once the lifespan has loaded it.
common_routes = APIRouter()
embedder_routes = APIRouter()
reranker_routes = APIRouter()


def create_model_server(part: ModelServerPart, ports: RagPorts) -> FastAPI:
    """Build the server for one model over the ports; it always serves the local model.

    The served name is the alias the clients send (``RAG_EMBEDDING_API_MODEL`` /
    ``RAG_RERANK_API_MODEL``); the weights come from ``RAG_EMBEDDING_MODEL`` /
    ``RAG_RERANK_MODEL``.

    Args:
        part: The one model this process serves, and so its one inference route.
        ports: Builds the in-process model when the lifespan starts.

    Returns:
        The app, its routes answering under ``DOMINO_RUN_HOST_PATH`` and at the root.

    Example:
        >>> create_model_server("reranker", RagPorts(settings, reranker=FakeReranker()))
    """
    app = FastAPI(
        title=f"rag-model-server ({part})",
        root_path=os.environ.get("DOMINO_RUN_HOST_PATH", ""),  # Domino's prefix
        lifespan=_lifespan,
    )
    app.state.part, app.state.ports = part, ports
    app.state.alias = _served_alias(part, ports.settings)
    app.include_router(common_routes)
    app.include_router(embedder_routes if part == "embedder" else reranker_routes)
    return app


def _served_alias(part: ModelServerPart, settings: RagSettings) -> str:
    if part == "embedder":
        return settings.embedding_api_model
    return settings.rerank_api_model


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    ports: RagPorts = app.state.ports
    async with ports:
        if app.state.part == "embedder":
            app.state.embedder = await ports.local_embedder()
        else:
            app.state.reranker = await ports.local_reranker()
        yield


@common_routes.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    """Answer while the process is up, naming the model it serves.

    Args:
        request: The HTTP request, for the app's part.

    Returns:
        ``{"status": "ok", "role": ...}``: ``embedder`` or ``reranker``, under
        the key the route has always used.
    """
    return {"status": "ok", "role": request.app.state.part}


@common_routes.get(MODEL_SERVER_MODEL_ROUTE)
async def model_status(name: str, request: Request) -> JSONResponse:
    """Answer OVMS's model status route for the named model.

    Args:
        name: The model the client asks about.
        request: The HTTP request, for the app's alias and part.

    Returns:
        200 naming the alias and part (under ``role``), or 404 when the name is
        not the alias.
    """
    alias = request.app.state.alias
    if name != alias:
        body = {"error": f"unknown model {name!r}, this server serves {alias!r}"}
        return JSONResponse(status_code=HTTPStatus.NOT_FOUND, content=body)
    content = {"name": alias, "role": request.app.state.part}
    return JSONResponse(status_code=HTTPStatus.OK, content=content)


@embedder_routes.post(MODEL_SERVER_EMBEDDINGS_PATH)
async def embeddings(body: OpenAIEmbeddingsRequest, request: Request) -> JSONResponse:
    """Embed the texts of the request, as OVMS does.

    Args:
        body: The model alias and the texts (one bare string reads as one text).
        request: The HTTP request, for the app's alias and embedder.

    Returns:
        One vector per text, or OVMS's 404 when the model is not the alias.
    """
    alias = request.app.state.alias
    if body.model != alias:
        return _unknown_model()
    vectors = await request.app.state.embedder.embed(body.input)
    return JSONResponse(embeddings_response_for(alias, vectors).model_dump())


@reranker_routes.post(MODEL_SERVER_RERANK_PATH)
async def rerank(body: CohereRerankRequest, request: Request) -> JSONResponse:
    """Rerank the documents of the request against its query, as OVMS does.

    Args:
        body: The model alias, the query, the documents and an optional top_n.
        request: The HTTP request, for the app's alias and reranker.

    Returns:
        The top_n documents (all when unset) by score, or OVMS's 404 when the
        model is not the alias.
    """
    if body.model != request.app.state.alias:
        return _unknown_model()
    passages = candidates_from_rerank(body)
    top_n = body.top_n or len(passages)
    ranked = await request.app.state.reranker.rerank(body.query, passages, top_n)
    return JSONResponse(rerank_response_for(ranked).model_dump())


def _unknown_model() -> JSONResponse:
    # What OVMS answers when ``model`` names no served model (checked in its source).
    return JSONResponse(
        status_code=HTTPStatus.NOT_FOUND, content={"error": UNKNOWN_MODEL_ERROR}
    )


# --- the answers, in the standard formats' shapes ---------------------------------


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``rag-model-server --serve embedder|reranker [--host H] [--port P]``.

    Serves the one model with uvicorn, on 0.0.0.0:8888 unless told otherwise, with
    ports built from the ``RAG_*`` settings.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status, 0, once the server stops.
    """
    parser = argparse.ArgumentParser(
        prog="rag-model-server", description="Serve one model."
    )
    parser.add_argument("--serve", choices=get_args(ModelServerPart), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ports = RagPorts(RagSettings())
    uvicorn.run(create_model_server(args.serve, ports), host=args.host, port=args.port)
    return 0
