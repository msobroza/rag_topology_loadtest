"""FastAPI service exposing the workflow, plus the ``rag-serve`` command.

Routes: ``POST /query`` (full RAG), ``POST /retrieve`` (no LLM call),
``GET /healthz``, ``GET /readyz``. Every response carries ``X-Request-ID``,
``X-RAG-Topology`` and ``X-RAG-Run-Label`` so Locust can tag runs; dependency
failures map to 503
(unavailable) or 504 (timeout). ``root_path`` follows ``DOMINO_RUN_HOST_PATH``, the
prefix Domino serves the app under, so the app answers with or without the
prefix on the wire and needs no proxy in front (unlike OVMS).

Example::

    uvicorn.run(create_app(RagPorts(RagSettings())), host="0.0.0.0", port=8888)
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import uuid
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Literal

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..adapters.json_log import log_json
from ..adapters.rag_ports import RagPorts
from ..application.workflow import RagWorkflow
from ..config.domino_rules import apply_domino_rules
from ..config.settings import RagSettings
from ..domain.errors import RagDependencyError
from ..domain.passages import RagMode, ScoredPassage
from ..domain.ports import RagDependencies

logger = logging.getLogger("rag_load_test")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    request_id: str
    topology: str
    run_label: str
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
    probes = (
        ("vector_store", deps.vector_store.ready),
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


# The handlers read what the factory left on ``app.state``: ``ports`` from the
# start, ``deps`` and ``run`` once the lifespan has opened the ports.
routes = APIRouter()


def create_app(ports: RagPorts) -> FastAPI:
    """Build the workflow app over the ports; they open and close with its lifespan.

    Args:
        ports: The ports the app serves with; their settings also name the
            topology and run label stamped on every response.

    Returns:
        The app, its routes answering under ``DOMINO_RUN_HOST_PATH`` and at the root.

    Example:
        >>> with TestClient(create_app(RagPorts(fake_settings))) as client:
        ...     client.post("/query", json={"question": "vacation days?"})
    """
    app = FastAPI(
        title="rag-load-test",
        root_path=os.environ.get("DOMINO_RUN_HOST_PATH", ""),
        lifespan=_lifespan,
    )
    app.state.ports = ports
    app.middleware("http")(_stamp_headers)
    app.exception_handler(RagDependencyError)(_dependency_error)
    app.include_router(routes)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    ports: RagPorts = app.state.ports
    settings = ports.settings
    async with ports:
        deps = await ports.workflow_dependencies()
        app.state.deps = deps
        app.state.run = traced(
            RagWorkflow(deps, settings.workflow_parameters()).run,
            enabled=settings.domino_tracing,
        )
        log_json(
            event="rag_startup",
            topology=settings.topology,
            run_label=settings.run_label,
            embedder=deps.embedder.model_name,
            llm=deps.chat_model.model_name,
        )
        yield


def _settings(request: Request) -> RagSettings:
    return request.app.state.ports.settings


async def _stamp_headers(request: Request, call_next: Callable[..., Any]) -> Any:
    request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    response = await call_next(request)
    settings = _settings(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-RAG-Topology"] = settings.topology
    response.headers["X-RAG-Run-Label"] = settings.run_label
    return response


async def _dependency_error(request: Request, exc: RagDependencyError) -> JSONResponse:
    status = (
        HTTPStatus.GATEWAY_TIMEOUT
        if exc.kind == "timeout"
        else HTTPStatus.SERVICE_UNAVAILABLE
    )
    error = f"{exc.component}_{exc.kind}"
    log_json(
        event="rag_request",
        request_id=request.state.request_id,
        status=status.value,
        error=error,
    )
    body = {"error": error, "detail": str(exc), "target": exc.target}
    return JSONResponse(status_code=status, content=body)


async def _answer(request: Request, body: QueryRequest, mode: RagMode) -> QueryResponse:
    settings = _settings(request)
    result = await request.app.state.run(body.question, mode=mode)
    log_json(
        event="rag_request",
        request_id=request.state.request_id,
        topology=settings.topology,
        run_label=settings.run_label,
        mode=mode,
        timings_ms=result.timings_ms,
        status=HTTPStatus.OK.value,
    )
    return QueryResponse(
        request_id=request.state.request_id,
        topology=settings.topology,
        run_label=settings.run_label,
        **result.model_dump(),
    )


@routes.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Answer the question with the full workflow, LLM call included.

    Args:
        body: The question.
        request: The HTTP request, for its request id and the app's state.

    Returns:
        The answer, the passages it used and the time each stage took.
    """
    return await _answer(request, body, "query")


@routes.post("/retrieve", response_model=QueryResponse)
async def retrieve(body: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve and rerank the passages for the question, without the LLM call.

    Args:
        body: The question.
        request: The HTTP request, for its request id and the app's state.

    Returns:
        The passages, no answer, and the time each stage took.
    """
    return await _answer(request, body, "retrieve")


@routes.get("/healthz")
async def healthz() -> dict[str, str]:
    """Answer while the process is up, whatever its dependencies.

    Returns:
        ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@routes.get("/readyz", response_model=ReadyResponse)
async def readyz(request: Request) -> JSONResponse:
    """Probe the store and the model ports.

    Args:
        request: The HTTP request, for the app's opened dependencies.

    Returns:
        200 with every check when all pass, else 503 with the same body.
    """
    checks = await readiness(request.app.state.deps)
    ready = all(check.ok for check in checks)
    body = ReadyResponse(status="ready" if ready else "not_ready", checks=checks)
    status = HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status, content=body.model_dump())


def main(argv: Sequence[str] | None = None) -> int:
    """``rag-serve [--host H] [--port P] [--domino]``; Domino needs 0.0.0.0:8888."""
    parser = argparse.ArgumentParser(
        prog="rag-serve", description="Serve the RAG workflow."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--domino",
        action="store_true",
        help="require model server URLs and default RAG_OVMS_AUTH to domino",
    )
    args = parser.parse_args(argv)
    try:
        settings = RagSettings()
        if args.domino:
            settings = apply_domino_rules(settings)
    except ValueError as exc:  # a settings mistake is a usage error, not a crash
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(create_app(RagPorts(settings)), host=args.host, port=args.port)
    return 0
