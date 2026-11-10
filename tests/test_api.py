"""API tests over fake dependencies: routes, headers, readiness, error mapping, CLI."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_load_test import api
from rag_load_test.adapters import INDEX
from rag_load_test.api import create_app, main, traced
from rag_load_test.fakes import (
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
    InMemoryVectorStore,
)
from rag_load_test.models import RagDependencyError, ScoredPassage
from rag_load_test.settings import RagSettings
from rag_load_test.workflow import RagDependencies

TOPOLOGY = "test-topo"
QUESTION = "How many paid vacation days do employees accrue each year?"
RERANK_URL = "http://ovms:8001/v3/rerank"
RESPONSE_FIELDS = {
    "request_id",
    "deployment",
    "question",
    "mode",
    "answer",
    "passages",
    "timings_ms",
}


class FailingReranker:
    """RerankerPort whose ``rerank`` raises a ``RagDependencyError`` of ``kind``."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        raise RagDependencyError("reranker", "boom", target=RERANK_URL, kind=self.kind)

    async def ready(self) -> tuple[bool, str]:
        return True, "failing-reranker"


class ExplodingProbeReranker:
    """RerankerPort whose readiness probe itself raises."""

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        return list(passages[:top_n])

    async def ready(self) -> tuple[bool, str]:
        raise RuntimeError("reranker probe exploded")


@pytest.fixture
def client_for(
    settings_factory: Callable[..., RagSettings],
) -> Callable[[RagDependencies], TestClient]:
    """``TestClient`` factory; use it as a context manager so the lifespan runs."""
    settings = settings_factory(topology=TOPOLOGY)

    def make(deps: RagDependencies) -> TestClient:
        return TestClient(create_app(settings, dependencies=deps))

    return make


@pytest.fixture
def client(
    client_for: Callable[[RagDependencies], TestClient], fake_deps: RagDependencies
) -> Iterator[TestClient]:
    with client_for(fake_deps) as test_client:
        yield test_client


def _checks_by_name(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {check["name"]: check for check in body["checks"]}


# --- /query and /retrieve --------------------------------------------------------


def test_query_returns_answer_and_headers(client: TestClient) -> None:
    response = client.post("/query", json={"question": QUESTION})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == RESPONSE_FIELDS
    assert body["deployment"] == TOPOLOGY
    assert body["question"] == QUESTION
    assert body["mode"] == "query"
    assert body["answer"].startswith("[fake-llm]")
    assert body["passages"]
    assert all(p["rerank_score"] is not None for p in body["passages"])
    assert body["timings_ms"]["total"] > 0
    assert response.headers["X-RAG-Topology"] == TOPOLOGY
    assert response.headers["X-Request-ID"] == body["request_id"]


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
    assert body["passages"]


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
    assert checks["vector_store"]["detail"].endswith(f"passages in {INDEX}")


def test_readyz_empty_store_is_not_ready(
    client_for: Callable[[RagDependencies], TestClient],
) -> None:
    deps = RagDependencies(
        FakeEmbedder(), InMemoryVectorStore(), FakeReranker(), FakeChatModel()
    )
    with client_for(deps) as test_client:
        response = test_client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    checks = _checks_by_name(body)
    assert checks["vector_store"]["ok"] is False
    assert checks["embedder"]["ok"] and checks["reranker"]["ok"]


def test_readyz_reports_probe_exception(
    client_for: Callable[[RagDependencies], TestClient], fake_deps: RagDependencies
) -> None:
    deps = dataclasses.replace(fake_deps, reranker=ExplodingProbeReranker())
    with client_for(deps) as test_client:
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
    client_for: Callable[[RagDependencies], TestClient],
    fake_deps: RagDependencies,
    kind: str,
    status: int,
    error: str,
) -> None:
    deps = dataclasses.replace(fake_deps, reranker=FailingReranker(kind))
    with client_for(deps) as test_client:
        response = test_client.post("/query", json={"question": QUESTION})

    assert response.status_code == status
    body = response.json()
    assert body["error"] == error
    assert body["target"] == RERANK_URL
    assert "boom" in body["detail"]
    assert response.headers["X-RAG-Topology"] == TOPOLOGY
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

    with caplog.at_level("WARNING", logger="rag_load_test"):
        assert traced(_run, enabled=True) is _run

    assert "dominodatalab[agents]" in caplog.text


# --- rag-serve ----------------------------------------------------------------


def test_main_parses_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_app = object()
    runs: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(api, "create_app", lambda settings: sentinel_app)
    monkeypatch.setattr(api.uvicorn, "run", lambda app, **kw: runs.append((app, kw)))

    assert main(["--host", "127.0.0.1", "--port", "9999"]) == 0

    assert runs == [(sentinel_app, {"host": "127.0.0.1", "port": 9999})]


def test_main_defaults_to_domino_app_port(monkeypatch: pytest.MonkeyPatch) -> None:
    runs: list[dict[str, Any]] = []
    monkeypatch.setattr(api, "create_app", lambda settings: object())
    monkeypatch.setattr(api.uvicorn, "run", lambda app, **kw: runs.append(kw))

    assert main([]) == 0

    assert runs == [{"host": "0.0.0.0", "port": 8888}]


def test_routes_answer_with_and_without_the_domino_prefix(
    monkeypatch: pytest.MonkeyPatch, settings_factory, fake_deps
) -> None:
    """FastAPI serves under DOMINO_RUN_HOST_PATH itself; no proxy layer needed."""
    monkeypatch.setenv("DOMINO_RUN_HOST_PATH", "/apps/abc123")
    app = create_app(settings_factory(), dependencies=fake_deps)
    with TestClient(app) as client:
        assert client.get("/apps/abc123/healthz").status_code == 200
        assert client.get("/healthz").status_code == 200
