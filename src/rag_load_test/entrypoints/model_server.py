"""A FastAPI model server speaking the OVMS contract (the FastAPI-vs-OVMS experiment).

Like an OVMS app it serves ONE model per process: ``--serve embedder`` exposes
``POST /v3/embeddings``, ``--serve reranker`` exposes ``POST /v3/rerank``. Both
answer ``GET /v3/models/{name}`` for the readiness probe of the ``ovms`` adapters.
Paths and bodies come from :mod:`.v3_contract`, the same module the adapters read;
like OVMS, a request for any model but this server's alias is a 404.
The workflow app cannot tell the two servers apart, so ``RAG_OVMS_*_URL`` can
point at either and the same load test compares FastAPI with OVMS.

Example:
    $ rag-model-server --serve reranker --port 8011
    $ RAG_TOPOLOGY=split-reranker RAG_OVMS_RERANK_URL=http://localhost:8011 rag-serve
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Literal

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from ..adapters.ovms.v3_contract import (
    EMBEDDINGS_PATH,
    MODEL_PROBE_ROUTE,
    RERANK_PATH,
    UNKNOWN_MODEL_ERROR,
    EmbeddingsRequest,
    RerankRequest,
    candidates_from_rerank,
    embeddings_response_for,
    embeddings_texts,
    rerank_response_for,
)
from ..adapters.rag_ports import RagPorts
from ..config.settings import RagSettings

Role = Literal["embedder", "reranker"]


# The handlers read what the factory left on ``app.state``: ``role``, ``alias``
# and ``ports`` from the start, the served model once the lifespan has loaded it.
common_routes = APIRouter()
embedder_routes = APIRouter()
reranker_routes = APIRouter()


def create_model_server(role: Role, ports: RagPorts) -> FastAPI:
    """Build the server for the role over the ports; it always serves the local model.

    The served name is the alias the clients use (``RAG_OVMS_EMBEDDINGS_MODEL`` /
    ``RAG_OVMS_RERANK_MODEL``); the weights come from ``RAG_EMBEDDER_MODEL`` /
    ``RAG_RERANKER_MODEL``.

    Args:
        role: The one model this process serves, and so its one inference route.
        ports: Builds the in-process model when the lifespan starts.

    Returns:
        The app, its routes answering under ``DOMINO_RUN_HOST_PATH`` and at the root.

    Example:
        >>> create_model_server("reranker", RagPorts(settings, reranker=FakeReranker()))
    """
    app = FastAPI(
        title=f"rag-model-server ({role})",
        root_path=os.environ.get("DOMINO_RUN_HOST_PATH", ""),  # Domino's prefix
        lifespan=_lifespan,
    )
    app.state.role, app.state.ports = role, ports
    app.state.alias = _served_alias(role, ports.settings)
    app.include_router(common_routes)
    app.include_router(embedder_routes if role == "embedder" else reranker_routes)
    return app


def _served_alias(role: Role, settings: RagSettings) -> str:
    if role == "embedder":
        return settings.ovms_embeddings_model
    return settings.ovms_rerank_model


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    ports: RagPorts = app.state.ports
    async with ports:
        if app.state.role == "embedder":
            app.state.embedder = await ports.local_embedder()
        else:
            app.state.reranker = await ports.local_reranker()
        yield


@common_routes.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    """Answer while the process is up, naming the role it serves.

    Args:
        request: The HTTP request, for the app's role.

    Returns:
        ``{"status": "ok", "role": ...}``.
    """
    return {"status": "ok", "role": request.app.state.role}


@common_routes.get(MODEL_PROBE_ROUTE)
async def model_status(name: str, request: Request) -> JSONResponse:
    """Answer the adapters' readiness probe for the named model.

    Args:
        name: The model the client asks about.
        request: The HTTP request, for the app's alias and role.

    Returns:
        200 naming the alias and role, or 404 when the name is not the alias.
    """
    alias = request.app.state.alias
    if name != alias:
        body = {"error": f"unknown model {name!r}, this server serves {alias!r}"}
        return JSONResponse(status_code=HTTPStatus.NOT_FOUND, content=body)
    content = {"name": alias, "role": request.app.state.role}
    return JSONResponse(status_code=HTTPStatus.OK, content=content)


@embedder_routes.post(EMBEDDINGS_PATH)
async def embeddings(body: EmbeddingsRequest, request: Request) -> JSONResponse:
    """Embed the texts of the request, as OVMS does.

    Args:
        body: The model alias and the texts.
        request: The HTTP request, for the app's alias and embedder.

    Returns:
        One vector per text, or OVMS's 404 when the model is not the alias.
    """
    alias = request.app.state.alias
    if body.model != alias:
        return _unknown_model()
    vectors = await request.app.state.embedder.embed(embeddings_texts(body))
    return JSONResponse(embeddings_response_for(alias, vectors).model_dump())


@reranker_routes.post(RERANK_PATH)
async def rerank(body: RerankRequest, request: Request) -> JSONResponse:
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
    parser.add_argument("--serve", choices=("embedder", "reranker"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ports = RagPorts(RagSettings())
    uvicorn.run(create_model_server(args.serve, ports), host=args.host, port=args.port)
    return 0
