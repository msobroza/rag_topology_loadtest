"""FastAPI service exposing the workflow, plus the ``qp-serve`` command.

Routes: ``POST /query`` (full RAG), ``POST /retrieve`` (no LLM call),
``GET /healthz``, ``GET /readyz``. Every response carries ``X-Request-ID`` and
``X-RAG-Run-Label``, and ``X-RAG-Topology`` when a topology alias was given
(``QP_TOPOLOGY_ALIAS``, which the deploy tool sets), so Locust can tag runs;
dependency failures map to 503 (unavailable) or 504 (timeout). ``root_path``
follows ``DOMINO_RUN_HOST_PATH``, the prefix Domino serves the app under, so the
app answers with or without the prefix on the wire and needs no proxy in front
(unlike OVMS).

Example:
    >>> app = create_app(CompositionRoot(WorkflowSettings()))
    >>> uvicorn.run(app, host="0.0.0.0", port=8888)
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
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..application.graph_check import WorkflowGraphError
from ..application.rag_dependencies import RagDependencies
from ..application.workflow_builder import build_workflow
from ..composition.composition_root import (
    CompositionRoot,
    check_workflow_graph,
    stage_context,
)
from ..composition.registered_names import check_registered_names
from ..config.workflow_settings import WorkflowSettings
from ..domain.conversation import check_history_conditions
from ..domain.errors import RagDependencyError
from ..domain.query_request import QueryRequest
from ..domain.retrieval.metadata_filter import (
    MetadataCondition,
    MetadataFieldInfo,
    check_conditions,
)
from ..domain.retrieval.passages import RagMode, ScoredPassage
from ..infrastructure.json_log import log_json

logger = logging.getLogger("query_processing")


class QueryResponse(BaseModel):
    """The workflow's answer, its passages, the time and tokens of each stage.

    ``tokens`` holds, under each stage whose LLM reported both counts, the
    ``prompt`` and ``completion`` tokens; ``{}`` when no stage did, as on a
    ``/retrieve`` that asks the LLM nothing.
    """

    request_id: str
    # The topology alias, as X-RAG-Topology; None when none was given.
    topology: str | None
    run_label: str
    question: str
    mode: RagMode
    answer: str | None
    passages: list[ScoredPassage]
    timings_ms: dict[str, float]
    tokens: dict[str, dict[str, int]]
    # The conditions applied: the client's, then those the stages added.
    metadata_filter: list[MetadataCondition]


class ReadyCheck(BaseModel):
    """One dependency probe of ``/readyz``: which dependency, whether it passed, why."""

    name: str
    ok: bool
    detail: str


class ReadyResponse(BaseModel):
    """The ``/readyz`` verdict and its checks; ``ready`` only when all checks pass."""

    status: Literal["ready", "not_ready"]
    checks: list[ReadyCheck]


def find_add_tracing() -> Callable[..., Any] | None:
    """Find Domino's ``add_tracing`` decorator factory.

    Returns:
        The factory from ``domino.aisystems.tracing`` or, failing that,
        ``domino.agents.tracing``; ``None`` without the SDK.
    """
    for module in ("domino.aisystems.tracing", "domino.agents.tracing"):
        try:
            return importlib.import_module(module).add_tracing
        except (ImportError, AttributeError):
            continue
    return None


def traced(fn: Callable[..., Any], *, enabled: bool) -> Callable[..., Any]:
    """Wrap ``fn`` with Domino agent tracing when enabled and installed.

    Args:
        fn: The callable to trace, recorded as the ``rag_query`` span.
        enabled: Whether tracing is asked for (``QP_DOMINO__TRACING``).

    Returns:
        The traced callable, or ``fn`` itself when tracing is off or the SDK is
        missing (the latter logs a warning).
    """
    if not enabled:
        return fn
    add_tracing = find_add_tracing()
    if add_tracing is None:
        logger.warning("QP_DOMINO__TRACING is on but dominodatalab[agents] is missing")
        return fn
    return add_tracing(name="rag_query", autolog_frameworks=["langchain"])(fn)


async def readiness(deps: RagDependencies) -> list[ReadyCheck]:
    """Probe the store and the model ports; a failing probe is a failed check.

    A probe that raises is reported as a failed check naming the exception, so
    ``/readyz`` never crashes.

    Args:
        deps: The opened components, whose vector stores, embedders and
            rerankers are probed.

    Returns:
        One check per probed component, named after it, in the order opened.
    """
    checks = []
    for name, probe in deps.ready_probes():
        try:
            ok, detail = await probe()
        except Exception as exc:  # /readyz reports failures, never crashes
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append(ReadyCheck(name=name, ok=ok, detail=detail))
    return checks


# The handlers read what the factory left on ``app.state``: ``composition`` from the
# start, ``deps`` and ``run`` once the lifespan has opened the ports.
routes = APIRouter()


def create_app(composition: CompositionRoot) -> FastAPI:
    """Build the workflow app; its ports open and close with the app's lifespan.

    Args:
        composition: Builds the ports the app serves with; its settings also name
            the run label stamped on every response, and the topology alias
            when one was given.

    Returns:
        The app, its routes answering under ``DOMINO_RUN_HOST_PATH`` and at the root.

    Example:
        >>> with TestClient(create_app(CompositionRoot(fake_settings))) as client:
        ...     client.post("/query", json={"question": "vacation days?"})
    """
    app = FastAPI(
        title="query-processing",
        root_path=os.environ.get("DOMINO_RUN_HOST_PATH", ""),
        lifespan=_lifespan,
    )
    app.state.composition = composition
    app.state.metadata_fields = declared_metadata_fields(composition.settings)
    app.middleware("http")(_stamp_headers)
    app.add_middleware(GraphRefusalInOneLine)
    app.exception_handler(RagDependencyError)(_dependency_error)
    app.include_router(routes)
    return app


class GraphRefusalInOneLine:
    """ASGI middleware: a graph refused at start-up reaches the server as one line.

    Starlette hands the server a failed start-up as the whole traceback, which
    the server logs. A graph refused once its components are open (a store that
    ranks no text under a lexical retrieval) is a mistake in the settings, not
    a crash, so the server gets its message alone. Any other failure keeps its
    traceback.

    Args:
        app: The app whose start-up this watches.

    Example:
        >>> app.add_middleware(GraphRefusalInOneLine)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass a request on; hold a failed start-up until its cause is known.

        Args:
            scope: The ASGI scope; only a ``lifespan`` one is watched.
            receive: The server's receive channel.
            send: The server's send channel.

        Raises:
            WorkflowGraphError: The graph was refused at start-up; re-raised
                once the server has its one-line message.
        """
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return
        # Held until the error that caused it is known, then sent once.
        failed: list[Message] = []

        async def hold_failure(message: Message) -> None:
            if message["type"] == "lifespan.startup.failed":
                failed.append(message)
            else:
                await send(message)

        try:
            await self.app(scope, receive, hold_failure)
        except WorkflowGraphError as exc:
            if failed:
                one_line = f"qp-serve: cannot start: {exc}".replace("\n", " ")
                failed[:] = [{**failed[0], "message": one_line}]
            raise
        finally:
            for message in failed:
                await send(message)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    composition: CompositionRoot = app.state.composition
    settings = composition.settings
    async with composition:
        deps = await composition.workflow_dependencies()
        app.state.deps = deps
        context = stage_context(settings)
        components = deps.components()
        orchestrator = composition.orchestrator()
        workflow = build_workflow(settings.graph, components, context, orchestrator)
        app.state.run = traced(workflow.run, enabled=settings.domino.tracing)
        log_json(
            event="rag_startup",
            topology_alias=settings.topology_alias,
            run_label=settings.run_label,
            orchestrator=settings.orchestrator,
            models=deps.model_names(),
        )
        yield


def _settings(request: Request) -> WorkflowSettings:
    return request.app.state.composition.settings


async def _stamp_headers(request: Request, call_next: Callable[..., Any]) -> Any:
    request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    response = await call_next(request)
    settings = _settings(request)
    response.headers["X-Request-ID"] = request.state.request_id
    # No alias given, no topology reported: the header is left out, not empty.
    if settings.topology_alias is not None:
        response.headers["X-RAG-Topology"] = settings.topology_alias
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


def declared_metadata_fields(settings: WorkflowSettings) -> list[MetadataFieldInfo]:
    """Return the metadata fields a request's filter can name.

    They are those the vector stores the graph's retrieve stages call declare
    under ``metadata_field_info``, each name once, in the order of the stages.

    Args:
        settings: The workflow settings: the graph and its components.

    Returns:
        The declared fields.

    Raises:
        ValueError: If two stores declare one field differently; the message
            names the field, both stores and both declarations.

    Example:
        >>> [f.name for f in declared_metadata_fields(settings)]
        ['topic']
    """
    # The retrieve stage type is the one that hands a request's filter to a
    # store; a plain call to a store states its own reads.
    stores = settings.vector_stores()
    called = dict.fromkeys(
        stage.call
        for name, stage in settings.graph.stages.items()
        if stage.type_name(name) == "retrieve" and stage.call in stores
    )
    by_name: dict[str, tuple[str, MetadataFieldInfo]] = {}
    for store in called:
        for field in stores[store].metadata_field_info:
            _refuse_another_declaration(by_name.get(field.name), store, field)
            by_name.setdefault(field.name, (store, field))
    return [field for _, field in by_name.values()]


def _refuse_another_declaration(
    first: tuple[str, MetadataFieldInfo] | None, store: str, field: MetadataFieldInfo
) -> None:
    # A request's condition on the field is checked against one declaration.
    if first is None or first[1] == field:
        return
    raise ValueError(
        f"the vector stores {first[0]!r} and {store!r} declare the metadata field "
        f"{field.name!r} differently ({first[1].model_dump()} and "
        f"{field.model_dump()}); expected one declaration of a field a request's "
        "filter can name"
    )


def _refuse_conditions_that_do_not_fit(request: Request, body: QueryRequest) -> None:
    # A condition the declared fields do not allow is the client's mistake: a
    # 422 whose detail names it, as for a body of the wrong shape. A history
    # message's conditions are checked too, though only the prompts read them.
    fields = request.app.state.metadata_fields
    try:
        check_conditions(body.metadata_filter, fields)
        check_history_conditions(body.conversational_history, fields)
    except ValueError as exc:
        raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


async def _answer(request: Request, body: QueryRequest, mode: RagMode) -> QueryResponse:
    settings = _settings(request)
    _refuse_conditions_that_do_not_fit(request, body)
    result = await request.app.state.run(
        body.question,
        mode=mode,
        conversational_history=body.conversational_history,
        metadata_filter=body.metadata_filter,
    )
    log_json(
        event="rag_request",
        request_id=request.state.request_id,
        topology_alias=settings.topology_alias,
        run_label=settings.run_label,
        mode=mode,
        timings_ms=result.timings_ms,
        tokens=result.tokens,
        status=HTTPStatus.OK.value,
    )
    return QueryResponse(
        request_id=request.state.request_id,
        topology=settings.topology_alias,
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
        The answer, the passages it used, the time each stage took and the
        tokens of each stage whose LLM reported its counts.
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
    """Run ``qp-serve [--host H] [--port P] [--domino]``; Domino needs 0.0.0.0:8888.

    ``--domino`` says the app runs as a Domino App: a component on the
    ``served`` backend, a model server of ours and so another App behind
    Domino's proxy, is then called with the run token; off Domino it is called
    with no key. A settings mistake, an unknown backend, API format or
    orchestrator and a graph that cannot run, or that the orchestrator does not
    run, included, exits as a usage error before any component opens, not a
    crash.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status, 0, once the server stops.
    """
    parser = argparse.ArgumentParser(
        prog="qp-serve", description="Serve the RAG workflow."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--domino",
        action="store_true",
        help="run as a Domino App: call the served components with the run token",
    )
    args = parser.parse_args(argv)
    try:
        settings = WorkflowSettings()
        check_registered_names(settings)
        check_workflow_graph(settings)
    except ValueError as exc:  # a settings mistake is a usage error, not a crash
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    composition = CompositionRoot(settings, on_domino=args.domino)
    uvicorn.run(create_app(composition), host=args.host, port=args.port)
    return 0
