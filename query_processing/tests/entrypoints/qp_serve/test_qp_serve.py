"""Workflow app tests, started from all-fake settings: routes, headers, readiness, error mapping, CLI."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient
from query_processing_core.retrieval.vector_store_record import ScoredVectorStoreRecord

from query_processing.adapters.reranker.base_reranker import BaseReranker
from query_processing.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.apps.uvicorn_settings import UvicornSettings
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.errors import RagDependencyError
from query_processing.entrypoints import qp_serve as api
from query_processing.entrypoints.qp_serve import create_app, main, traced
from rag_components import FAKE_BACKENDS

TOPOLOGY_ALIAS = "split-all"  # a label: every backend stays fake
RUN_LABEL = "split-all-test-tier"
QUESTION = "How many paid vacation days do employees accrue each year?"
RERANK_URL = "http://ovms:8001/v3/rerank"
# What main passes Uvicorn besides the address when no QP_UVICORN_* is set.
UVICORNS_OWN = UvicornSettings(_env_file=None).run_arguments()
RESPONSE_FIELDS = {
    "request_id",
    "topology",
    "run_label",
    "question",
    "mode",
    "answer",
    "records",
    "timings_ms",
    "tokens",
    "metadata_filter",
}


class FailingReranker(BaseReranker):
    """BaseReranker whose ``rerank`` raises a ``RagDependencyError`` of ``kind``."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    async def rerank(
        self, query: str, passages: Sequence[ScoredVectorStoreRecord], top_n: int
    ) -> list[ScoredVectorStoreRecord]:
        raise RagDependencyError("reranker", "boom", target=RERANK_URL, kind=self.kind)

    async def ready(self) -> tuple[bool, str]:
        return True, "failing-reranker"


class ExplodingProbeReranker(BaseReranker):
    """BaseReranker whose readiness probe itself raises."""

    async def rerank(
        self, query: str, passages: Sequence[ScoredVectorStoreRecord], top_n: int
    ) -> list[ScoredVectorStoreRecord]:
        return list(passages[:top_n])

    async def ready(self) -> tuple[bool, str]:
        raise RuntimeError("reranker probe exploded")


@pytest.fixture
def client_for(
    settings_factory: Callable[..., WorkflowSettings],
) -> Callable[..., TestClient]:
    """The all-fake app, with any port replaced by keyword (``reranker=...``).

    Use the client as a context manager so the lifespan opens the ports.
    """
    # Built, not model_copy'd: only construction validates the labels.
    settings = settings_factory(
        topology_alias=TOPOLOGY_ALIAS,
        run_label=RUN_LABEL,
        components__embedder__backend="fake",
        components__reranker__backend="fake",
        components__llm__backend="fake",
        components__vector_store__backend="fake",
    )

    def make(**doubles: Any) -> TestClient:
        return TestClient(create_app(CompositionRoot(settings, **doubles)))

    return make


@pytest.fixture
def client(client_for: Callable[..., TestClient]) -> Iterator[TestClient]:
    with client_for() as test_client:
        yield test_client


def _checks_by_name(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {check["name"]: check for check in body["checks"]}


# --- /query and /retrieve --------------------------------------------------------


def test_query_returns_answer_and_headers(client: TestClient) -> None:
    response = client.post("/query", json={"question": QUESTION})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == RESPONSE_FIELDS
    assert body["topology"] == TOPOLOGY_ALIAS
    assert body["run_label"] == RUN_LABEL
    assert body["question"] == QUESTION
    assert body["mode"] == "query"
    assert body["answer"].startswith("[fake-llm]")
    assert body["records"]
    assert all(p["rerank_score"] is not None for p in body["records"])
    assert body["timings_ms"]["total"] > 0
    assert response.headers["X-RAG-Topology"] == TOPOLOGY_ALIAS
    assert response.headers["X-RAG-Run-Label"] == RUN_LABEL
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_unset_run_label_reports_each_component_with_its_backend(
    fake_settings: WorkflowSettings,
) -> None:
    with TestClient(create_app(CompositionRoot(fake_settings))) as test_client:
        response = test_client.post("/retrieve", json={"question": QUESTION})

    default = "embedder=fake+reranker=fake+llm=fake+vector_store=fake"
    assert response.json()["run_label"] == default
    assert response.headers["X-RAG-Run-Label"] == default


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_with_no_topology_alias_no_topology_is_reported(
    fake_settings: WorkflowSettings, path: str
) -> None:
    with TestClient(create_app(CompositionRoot(fake_settings))) as test_client:
        response = test_client.post(path, json={"question": QUESTION})
        readyz = test_client.get("/readyz")

    assert response.status_code == 200, response.text
    assert response.json()["topology"] is None
    for answered in (response, readyz):
        assert "X-RAG-Topology" not in answered.headers
        assert answered.headers["X-RAG-Run-Label"] == fake_settings.run_label


def test_any_topology_alias_is_reported_as_it_is_given(
    settings_factory: Callable[..., WorkflowSettings],
) -> None:
    # The deploy settings define the aliases: a fourth one names itself.
    settings = settings_factory(topology_alias="split-embedder", **FAKE_BACKENDS)

    with TestClient(create_app(CompositionRoot(settings))) as test_client:
        response = test_client.post("/query", json={"question": QUESTION})

    assert response.status_code == 200, response.text
    assert response.headers["X-RAG-Topology"] == "split-embedder"
    assert response.json()["topology"] == "split-embedder"
    assert response.json()["answer"].startswith("[fake-llm]")  # nothing moved


def test_supplied_request_id_is_echoed(client: TestClient) -> None:
    response = client.post(
        "/query", json={"question": QUESTION}, headers={"X-Request-ID": "req-123"}
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Request-ID"] == "req-123"
    assert response.json()["request_id"] == "req-123"


def test_retrieve_skips_generation(client: TestClient) -> None:
    response = client.post("/retrieve", json={"question": QUESTION})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "retrieve"
    assert body["answer"] is None
    assert body["timings_ms"]["generate"] == 0.0
    assert body["records"]
    assert body["tokens"] == {}


def test_a_query_answer_carries_its_tokens(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO", logger="query_processing"):
        body = client.post("/query", json={"question": QUESTION}).json()

    assert list(body["tokens"]) == ["generate"]
    assert body["tokens"]["generate"]["completion"] > 0
    [logged] = [
        json.loads(r.getMessage())
        for r in caplog.records
        if '"event": "rag_request"' in r.getMessage()
    ]
    assert logged["tokens"] == body["tokens"]


@pytest.mark.parametrize("path", ["/query", "/retrieve"])
def test_empty_question_is_rejected(client: TestClient, path: str) -> None:
    assert client.post(path, json={"question": ""}).status_code == 422


# --- health and readiness --------------------------------------------------------


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_all_checks_ok(client: TestClient) -> None:
    response = client.get("/readyz")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready"
    checks = _checks_by_name(body)
    assert set(checks) == {"vector_store", "embedder", "reranker"}
    assert all(check["ok"] for check in checks.values())
    assert checks["vector_store"]["detail"] == "200 passages in memory"


def test_readyz_empty_store_is_not_ready(
    client_for: Callable[..., TestClient],
) -> None:
    with client_for(vector_store=InMemoryVectorStore()) as test_client:
        response = test_client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    checks = _checks_by_name(body)
    assert checks["vector_store"]["ok"] is False
    assert checks["embedder"]["ok"]
    assert checks["reranker"]["ok"]


def test_readyz_reports_probe_exception(
    client_for: Callable[..., TestClient],
) -> None:
    with client_for(reranker=ExplodingProbeReranker()) as test_client:
        response = test_client.get("/readyz")

    assert response.status_code == 503
    checks = _checks_by_name(response.json())
    assert checks["reranker"]["ok"] is False
    assert "RuntimeError: reranker probe exploded" in checks["reranker"]["detail"]


# --- dependency errors -----------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "status", "error"),
    [
        ("unavailable", 503, "reranker_unavailable"),
        ("timeout", 504, "reranker_timeout"),
    ],
)
def test_dependency_error_maps_to_status(
    client_for: Callable[..., TestClient],
    kind: str,
    status: int,
    error: str,
) -> None:
    with client_for(reranker=FailingReranker(kind)) as test_client:
        response = test_client.post("/query", json={"question": QUESTION})

    assert response.status_code == status
    body = response.json()
    assert body["error"] == error
    assert body["target"] == RERANK_URL
    assert "boom" in body["detail"]
    assert response.headers["X-RAG-Topology"] == TOPOLOGY_ALIAS
    assert response.headers["X-RAG-Run-Label"] == RUN_LABEL
    assert response.headers["X-Request-ID"]


# --- traced() ------------------------------------------------------------------


def _run(question: str, *, mode: str = "query") -> str:
    return f"{mode}:{question}"


def test_traced_disabled_returns_function_unchanged() -> None:
    assert traced(_run, enabled=False) is _run


def test_traced_wraps_with_domino_add_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def add_tracing(
        **kwargs: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        calls.append(kwargs)

        def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(*args: Any, **kw: Any) -> tuple[str, Any]:
                return "traced", fn(*args, **kw)

            return wrapped

        return decorate

    monkeypatch.setattr(api, "find_add_tracing", lambda: add_tracing)

    wrapped = traced(_run, enabled=True)

    assert wrapped is not _run
    assert calls == [{"name": "rag_query", "autolog_frameworks": ["langchain"]}]
    assert wrapped("q", mode="retrieve") == ("traced", "retrieve:q")


def test_traced_without_sdk_returns_function_unchanged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(api, "find_add_tracing", lambda: None)

    with caplog.at_level("WARNING", logger="query_processing"):
        assert traced(_run, enabled=True) is _run

    assert "dominodatalab[agents]" in caplog.text


# --- qp-serve ----------------------------------------------------------------


def test_main_parses_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_app = object()
    runs: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(api, "create_app", lambda composition: sentinel_app)
    monkeypatch.setattr(api.uvicorn, "run", lambda app, **kw: runs.append((app, kw)))

    assert main(["--host", "127.0.0.1", "--port", "9999"]) == 0

    assert runs == [(sentinel_app, {"host": "127.0.0.1", "port": 9999, **UVICORNS_OWN})]


def test_main_defaults_to_domino_app_port(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[dict[str, Any]] = []
    monkeypatch.setattr(api, "create_app", lambda composition: object())
    monkeypatch.setattr(api.uvicorn, "run", lambda app, **kw: runs.append(kw))

    assert main([]) == 0

    assert runs == [{"host": "0.0.0.0", "port": 8888, **UVICORNS_OWN}]


def _capture_served_ports(monkeypatch: pytest.MonkeyPatch) -> list[CompositionRoot]:
    served: list[CompositionRoot] = []

    def record(composition: CompositionRoot) -> object:
        served.append(composition)
        return object()

    monkeypatch.setattr(api, "create_app", record)
    monkeypatch.setattr(api.uvicorn, "run", lambda app, **kw: None)
    return served


def test_main_domino_refuses_a_model_server_component_without_its_url(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QP_COMPONENTS__RERANKER__BACKEND", "served")
    monkeypatch.delenv("QP_COMPONENTS__RERANKER__SERVED__URL", raising=False)
    served = _capture_served_ports(monkeypatch)

    with pytest.raises(SystemExit) as exited:
        main(["--domino"])

    assert exited.value.code == 2
    assert "needs QP_COMPONENTS__RERANKER__SERVED__URL" in capsys.readouterr().err
    assert served == []


def test_main_ends_on_invalid_settings_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The embedder on the served backend, and no URL is set.
    monkeypatch.setenv("QP_COMPONENTS__EMBEDDER__BACKEND", "served")
    monkeypatch.delenv("QP_COMPONENTS__EMBEDDER__SERVED__URL", raising=False)
    served = _capture_served_ports(monkeypatch)

    with pytest.raises(SystemExit) as exited:
        main([])

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "QP_COMPONENTS__EMBEDDER__BACKEND='served' needs" in err
    assert "Traceback" not in err
    assert served == []


def test_routes_answer_with_and_without_the_domino_prefix(
    monkeypatch: pytest.MonkeyPatch, fake_settings: WorkflowSettings
) -> None:
    """FastAPI serves under DOMINO_RUN_HOST_PATH itself; no proxy layer needed."""
    monkeypatch.setenv("DOMINO_RUN_HOST_PATH", "/apps/abc123")
    app = create_app(CompositionRoot(fake_settings))
    with TestClient(app) as client:
        assert client.get("/apps/abc123/healthz").status_code == 200
        assert client.get("/healthz").status_code == 200
