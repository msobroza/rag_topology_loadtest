"""A FastAPI model server speaking the OVMS contract (the FastAPI-vs-OVMS experiment).

Like an OVMS app it serves ONE model per process: ``--serve embedder`` exposes
``POST /v3/embeddings``, ``--serve reranker`` exposes ``POST /v3/rerank``. Both
answer ``GET /v3/models/{name}`` for the readiness probe of the ``ovms`` adapters.
The workflow app cannot tell the two servers apart, so ``RAG_OVMS_*_URL`` can
point at either and the same load test compares FastAPI with OVMS.

Example::

    rag-model-server --serve reranker --port 8011
    RAG_OVMS_RERANK_URL=http://localhost:8011 RAG_RERANKER_BACKEND=ovms rag-serve
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .adapters import CrossEncoderReranker, SentenceTransformerEmbedder, model_executor
from .models import EmbedderPort, RerankerPort, ScoredPassage
from .settings import RagSettings

Role = Literal["embedder", "reranker"]


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | list[str]


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = None


def create_model_server(
    role: Role,
    settings: RagSettings | None = None,
    *,
    embedder: EmbedderPort | None = None,
    reranker: RerankerPort | None = None,
) -> FastAPI:
    """Build the server for ``role``; inject ``embedder``/``reranker`` in tests.

    The served name is the alias the clients use (``RAG_OVMS_EMBEDDINGS_MODEL`` /
    ``RAG_OVMS_RERANK_MODEL``); the weights come from ``RAG_EMBEDDER_MODEL`` /
    ``RAG_RERANKER_MODEL``.
    """
    settings = settings or RagSettings()
    alias = (
        settings.ovms_embeddings_model
        if role == "embedder"
        else settings.ovms_rerank_model
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        executor = model_executor()
        if role == "embedder":
            app.state.embedder = (
                embedder
                or SentenceTransformerEmbedder.from_pretrained(
                    settings.embedder_model, executor
                )
            )
        else:
            app.state.reranker = reranker or CrossEncoderReranker.from_pretrained(
                settings.reranker_model, executor
            )
        try:
            yield
        finally:
            executor.shutdown(wait=False)

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

    @app.get("/v3/models/{name}")
    async def model_status(name: str) -> JSONResponse:
        if name != alias:
            body = {"error": f"unknown model {name!r}, this server serves {alias!r}"}
            return JSONResponse(status_code=404, content=body)
        return JSONResponse(status_code=200, content={"name": alias, "role": role})

    if role == "embedder":

        @app.post("/v3/embeddings")
        async def embeddings(
            body: EmbeddingsRequest, request: Request
        ) -> dict[str, Any]:
            texts = [body.input] if isinstance(body.input, str) else body.input
            vectors = await request.app.state.embedder.embed(texts)
            data = [
                {"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(vectors)
            ]
            return {"object": "list", "model": body.model, "data": data}

    else:

        @app.post("/v3/rerank")
        async def rerank(body: RerankRequest, request: Request) -> dict[str, Any]:
            passages = [
                ScoredPassage(id=str(i), text=text, retrieval_score=0.0)
                for i, text in enumerate(body.documents)
            ]
            top_n = body.top_n or len(passages)
            ranked = await request.app.state.reranker.rerank(
                body.query, passages, top_n
            )
            results = [
                {"index": int(p.id), "relevance_score": p.rerank_score} for p in ranked
            ]
            return {"results": results}

    return app


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
    uvicorn.run(
        create_model_server(args.serve, RagSettings()), host=args.host, port=args.port
    )
    return 0
