"""A FastAPI model server speaking the OVMS contract (the FastAPI-vs-OVMS experiment).

Like an OVMS app it serves ONE model per process: ``--serve embedder`` exposes
``POST /v3/embeddings``, ``--serve reranker`` exposes ``POST /v3/rerank``. Both
answer ``GET /v3/models/{name}`` for the readiness probe of the ``ovms`` adapters.
Paths and bodies come from :mod:`.v3_contract`, the same module the adapters read;
like OVMS, a request for any model but this server's alias is a 404.
The workflow app cannot tell the two servers apart, so ``RAG_OVMS_*_URL`` can
point at either and the same load test compares FastAPI with OVMS.

Example::

    rag-model-server --serve reranker --port 8011
    RAG_TOPOLOGY=split-reranker RAG_OVMS_RERANK_URL=http://localhost:8011 rag-serve
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..adapters.rag_ports import RagPorts
from ..adapters.v3_contract import (
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
from ..config.settings import RagSettings

Role = Literal["embedder", "reranker"]


def create_model_server(role: Role, ports: RagPorts) -> FastAPI:
    """Build the server for ``role`` over ``ports``; it always serves the local model.

    The served name is the alias the clients use (``RAG_OVMS_EMBEDDINGS_MODEL`` /
    ``RAG_OVMS_RERANK_MODEL``); the weights come from ``RAG_EMBEDDER_MODEL`` /
    ``RAG_RERANKER_MODEL``.

    Example::

        create_model_server("reranker", RagPorts(settings, reranker=FakeReranker()))
    """
    settings = ports.settings
    alias = (
        settings.ovms_embeddings_model
        if role == "embedder"
        else settings.ovms_rerank_model
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with ports:
            if role == "embedder":
                app.state.embedder = await ports.local_embedder()
            else:
                app.state.reranker = await ports.local_reranker()
            yield

    app = FastAPI(
        title=f"rag-model-server ({role})",
        root_path=os.environ.get(
            "DOMINO_RUN_HOST_PATH", ""
        ),  # served under Domino's prefix
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "role": role}

    @app.get(MODEL_PROBE_ROUTE)
    async def model_status(name: str) -> JSONResponse:
        if name != alias:
            body = {"error": f"unknown model {name!r}, this server serves {alias!r}"}
            return JSONResponse(status_code=404, content=body)
        return JSONResponse(status_code=200, content={"name": alias, "role": role})

    if role == "embedder":

        @app.post(EMBEDDINGS_PATH)
        async def embeddings(body: EmbeddingsRequest, request: Request) -> JSONResponse:
            if body.model != alias:
                return _unknown_model()
            texts = embeddings_texts(body)
            vectors = await request.app.state.embedder.embed(texts)
            return JSONResponse(embeddings_response_for(alias, vectors).model_dump())

    else:

        @app.post(RERANK_PATH)
        async def rerank(body: RerankRequest, request: Request) -> JSONResponse:
            if body.model != alias:
                return _unknown_model()
            passages = candidates_from_rerank(body)
            top_n = body.top_n or len(passages)
            ranked = await request.app.state.reranker.rerank(
                body.query, passages, top_n
            )
            return JSONResponse(rerank_response_for(ranked).model_dump())

    return app


def _unknown_model() -> JSONResponse:
    # What OVMS answers when ``model`` names no served model (checked in its source).
    return JSONResponse(status_code=404, content={"error": UNKNOWN_MODEL_ERROR})


def main(argv: Sequence[str] | None = None) -> int:
    """``rag-model-server --serve embedder|reranker [--host H] [--port P]``."""
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
