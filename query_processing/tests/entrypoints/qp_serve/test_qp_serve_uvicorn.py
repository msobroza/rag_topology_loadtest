"""``qp-serve`` under Uvicorn's settings: one worker serves the app object, more the factory.

A worker builds its app by ``workflow_app_from_environment``, from the
settings and what ``main`` exported for it: the app it builds must answer as
the one ``main`` builds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from uvicorn.importer import import_from_string

from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints import qp_serve
from rag_components import FAKE_BACKEND_VARIABLES

Run = tuple[tuple[Any, ...], dict[str, Any]]


@pytest.fixture
def fake_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every component of the tests' workflow file on its fake backend."""
    for variable, value in FAKE_BACKEND_VARIABLES.items():
        monkeypatch.setenv(variable, value)


def create_app_as_main_does() -> FastAPI:
    """The app ``main`` builds and serves with one worker."""
    return qp_serve.create_app(CompositionRoot(WorkflowSettings()))


@pytest.mark.usefixtures("fake_backends")
@pytest.mark.parametrize(
    "build", [create_app_as_main_does, qp_serve.workflow_app_from_environment]
)
def test_the_app_object_and_the_factory_answer_healthz_alike(
    build: Callable[[], FastAPI],
) -> None:
    with TestClient(build()) as client:
        healthz = client.get("/healthz")
        ready = client.get("/readyz")

    assert (healthz.status_code, healthz.json()) == (200, {"status": "ok"})
    assert ready.status_code == 200, ready.text


@pytest.mark.parametrize(("exported", "on_domino"), [("1", True), ("0", False)])
def test_the_factory_runs_as_a_domino_app_when_main_was_told_so(
    monkeypatch: pytest.MonkeyPatch, exported: str, on_domino: bool
) -> None:
    built: list[dict[str, Any]] = []
    monkeypatch.setattr(
        qp_serve, "CompositionRoot", lambda settings, **kw: built.append(kw)
    )
    monkeypatch.setattr(qp_serve, "create_app", lambda composition: FastAPI())
    monkeypatch.setenv("QP_SERVE_DOMINO", exported)

    qp_serve.workflow_app_from_environment()

    assert built == [{"on_domino": on_domino}]


def serve_recording(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[Run]:
    """Run ``main`` with Uvicorn replaced; return what Uvicorn was asked."""
    runs: list[Run] = []
    monkeypatch.setattr(qp_serve, "create_app", lambda composition: "the app")
    monkeypatch.setattr(qp_serve.uvicorn, "run", lambda *a, **k: runs.append((a, k)))
    assert qp_serve.main(argv) == 0
    return runs


def test_main_passes_uvicorns_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QP_UVICORN_LIMIT_CONCURRENCY", "64")
    monkeypatch.setenv("QP_UVICORN_TIMEOUT_KEEP_ALIVE_S", "30")

    [(args, kwargs)] = serve_recording(monkeypatch, [])

    assert args == ("the app",)
    assert kwargs == {
        "host": "0.0.0.0",
        "port": 8888,
        "workers": 1,
        "limit_concurrency": 64,
        "backlog": 2048,
        "timeout_keep_alive": 30,
    }


def test_main_with_two_workers_serves_the_factory_and_exports_domino(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_UVICORN_WORKERS", "2")
    # A stale value, which main replaces; restored when the test ends.
    monkeypatch.setenv("QP_SERVE_DOMINO", "0")

    [(args, kwargs)] = serve_recording(monkeypatch, ["--domino"])

    [factory] = args
    assert import_from_string(factory) is qp_serve.workflow_app_from_environment
    assert (kwargs["factory"], kwargs["workers"]) == (True, 2)
    assert qp_serve.os.environ["QP_SERVE_DOMINO"] == "1"


def test_main_ends_on_an_invalid_uvicorn_setting_as_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QP_UVICORN_WORKERS", "0")
    monkeypatch.setattr(qp_serve.uvicorn, "run", lambda *a, **k: pytest.fail("ran"))

    with pytest.raises(SystemExit) as stopped:
        qp_serve.main([])

    assert stopped.value.code == 2
    assert "workers" in capsys.readouterr().err
