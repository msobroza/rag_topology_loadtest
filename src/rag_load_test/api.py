"""FastAPI service exposing the workflow, plus the ``rag-serve`` command.

Routes: ``POST /query`` (full RAG), ``POST /retrieve`` (no LLM call),
``GET /healthz``, ``GET /readyz``. Every response carries ``X-Request-ID`` and
``X-RAG-Topology`` so Locust can tag runs; dependency failures map to 503
(unavailable) or 504 (timeout). ``root_path`` follows ``DOMINO_RUN_HOST_PATH``, the
prefix Domino serves the app under, so the app answers with or without the
prefix on the wire and needs no proxy in front (unlike OVMS).

Example::

    uvicorn.run(create_app(RagSettings()), host="0.0.0.0", port=8888)
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .adapters import INDEX, model_executor
from .models import RagDependencyError, RagMode, ScoredPassage
from .ovms import TIMEOUT_S
from .settings import RagSettings
from .workflow import RagDependencies, RagWorkflow, build_dependencies

logger = logging.getLogger("rag_load_test")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    request_id: str
    deployment: str
    question: str
    mode: RagMode
    answer: str | None
    passages: list[ScoredPassage]
    timings_ms: dict[str, float]


class ReadyCheck(BaseModel):
    name: str
    ok: bool
    detail: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: list[ReadyCheck]


def log_json(**fields: Any) -> None:
    """One JSON object per log line (the app's only log format)."""
    logger.info(json.dumps(fields, default=str))


def find_add_tracing() -> Callable[..., Any] | None:
    """Domino's ``add_tracing`` decorator factory, or ``None`` without the SDK."""
    for module in ("domino.aisystems.tracing", "domino.agents.tracing"):
        try:
            return importlib.import_module(module).add_tracing
        except (ImportError, AttributeError):
            continue
    return None


def traced(fn: Callable[..., Any], *, enabled: bool) -> Callable[..., Any]:
    """Wrap ``fn`` with Domino agent tracing when enabled and installed."""
    if not enabled:
        return fn
    add_tracing = find_add_tracing()
    if add_tracing is None:
        logger.warning("RAG_DOMINO_TRACING is on but dominodatalab[agents] is missing")
        return fn
    return add_tracing(name="rag_query", autolog_frameworks=["langchain"])(fn)


async def readiness(deps: RagDependencies) -> list[ReadyCheck]:
    """Probe the store and the model ports; a failing probe is a failed check."""

    async def store() -> tuple[bool, str]:
        count = await deps.vector_store.count()
        return count > 0, f"{count} passages in {INDEX}"

    probes = (
        ("vector_store", store),
        ("embedder", deps.embedder.ready),
        ("reranker", deps.reranker.ready),
    )
    checks = []
    for name, probe in probes:
        try:
            ok, detail = await probe()
        except Exception as exc:  # /readyz reports failures, never crashes
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append(ReadyCheck(name=name, ok=ok, detail=detail))
    return checks


def create_app(
    settings: RagSettings | None = None, *, dependencies: RagDependencies | None = None
) -> FastAPI:
    """Build the app; pass ``dependencies`` to skip ``build_dependencies`` (tests).

    Example::

        with TestClient(create_app(settings, dependencies=fake_deps)) as client:
            client.post("/query", json={"question": "vacation days?"})
    """
    settings = settings or RagSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        executor = model_executor()
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            deps = dependencies or build_dependencies(
                settings, client=client, executor=executor
            )
            app.state.deps = deps
            app.state.run = traced(
                RagWorkflow(deps).run, enabled=settings.domino_tracing
            )
            log_json(
                event="rag_startup",
                topology=settings.topology,
                embedder=deps.embedder.model_name,
                llm=deps.chat_model.model_name,
            )
            try:
                yield
            finally:
                executor.shutdown(wait=False)

    app = FastAPI(
        title="rag-load-test",
        root_path=os.environ.get("DOMINO_RUN_HOST_PATH", ""),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def stamp_headers(request: Request, call_next: Callable[..., Any]) -> Any:
        request.state.request_id = (
            request.headers.get("X-Request-ID") or uuid.uuid4().hex
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-RAG-Topology"] = settings.topology
        return response

    @app.exception_handler(RagDependencyError)
    async def dependency_error(
        request: Request, exc: RagDependencyError
    ) -> JSONResponse:
        status = 504 if exc.kind == "timeout" else 503
        error = f"{exc.component}_{exc.kind}"
        log_json(
            event="rag_request",
            request_id=request.state.request_id,
            status=status,
            error=error,
        )
        body = {"error": error, "detail": str(exc), "target": exc.target}
        return JSONResponse(status_code=status, content=body)

    async def answer(
        request: Request, body: QueryRequest, mode: RagMode
    ) -> QueryResponse:
        result = await request.app.state.run(body.question, mode=mode)
        log_json(
            event="rag_request",
            request_id=request.state.request_id,
            topology=settings.topology,
            mode=mode,
            timings_ms=result.timings_ms,
            status=200,
        )
        return QueryResponse(
            request_id=request.state.request_id,
            deployment=settings.topology,
            **result.model_dump(),
        )

    @app.post("/query", response_model=QueryResponse)
    async def query(body: QueryRequest, request: Request) -> QueryResponse:
        return await answer(request, body, "query")

    @app.post("/retrieve", response_model=QueryResponse)
    async def retrieve(body: QueryRequest, request: Request) -> QueryResponse:
        return await answer(request, body, "retrieve")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=ReadyResponse)
    async def readyz(request: Request) -> JSONResponse:
        checks = await readiness(request.app.state.deps)
        ready = all(check.ok for check in checks)
        body = ReadyResponse(status="ready" if ready else "not_ready", checks=checks)
        return JSONResponse(
            status_code=200 if ready else 503, content=body.model_dump()
        )

    return app


def main(argv: Sequence[str] | None = None) -> int:
    """``rag-serve [--host H] [--port P]``; Domino apps must listen on 0.0.0.0:8888."""
    parser = argparse.ArgumentParser(
        prog="rag-serve", description="Serve the RAG workflow."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(create_app(RagSettings()), host=args.host, port=args.port)
    return 0
