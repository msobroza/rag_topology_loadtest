"""``qp-model-server`` under Uvicorn's settings: one worker serves the app object, more the factory.

A worker builds its app by ``model_server_from_environment``, from the settings
and ``QP_MODEL_SERVER_COMPONENT``, which ``main`` exports from ``--serve``.
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
from query_processing.entrypoints import qp_model_server

COMPONENT_VARIABLE = "QP_MODEL_SERVER_COMPONENT"
Run = tuple[tuple[Any, ...], dict[str, Any]]


def create_server_as_main_does() -> FastAPI:
    """The app ``main`` builds and serves with one worker, for ``--serve reranker``."""
    composition = CompositionRoot(WorkflowSettings())
    return qp_model_server.create_model_server("reranker", composition)


@pytest.mark.parametrize(
    "build",
    [create_server_as_main_does, qp_model_server.model_server_from_environment],
)
def test_the_app_object_and_the_factory_answer_healthz_alike(
    monkeypatch: pytest.MonkeyPatch, build: Callable[[], FastAPI]
) -> None:
    monkeypatch.setenv(COMPONENT_VARIABLE, "reranker")

    # No lifespan: the model is not loaded, and /healthz needs none.
    healthz = TestClient(build()).get("/healthz")

    assert (healthz.status_code, healthz.json()) == (
        200,
        {"status": "ok", "role": "reranker"},
    )


def test_the_factory_refuses_an_unset_component_naming_its_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(COMPONENT_VARIABLE, raising=False)

    with pytest.raises(ValueError, match=COMPONENT_VARIABLE):
        qp_model_server.model_server_from_environment()


def serve_recording(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[Run]:
    """Run ``main`` with Uvicorn replaced; return what Uvicorn was asked."""
    runs: list[Run] = []
    monkeypatch.setattr(
        qp_model_server, "create_model_server", lambda name, composition: name
    )
    monkeypatch.setattr(
        qp_model_server.uvicorn, "run", lambda *a, **k: runs.append((a, k))
    )
    assert qp_model_server.main(argv) == 0
    return runs


def test_main_passes_uvicorns_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QP_UVICORN_BACKLOG", "512")

    [(args, kwargs)] = serve_recording(monkeypatch, ["--serve", "embedder"])

    assert args == ("embedder",)
    assert kwargs == {
        "host": "0.0.0.0",
        "port": 8888,
        "workers": 1,
        "limit_concurrency": None,
        "backlog": 512,
        "timeout_keep_alive": 5,
    }


def test_main_with_two_workers_serves_the_factory_and_exports_the_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_UVICORN_WORKERS", "2")
    # A stale value, which main replaces; restored when the test ends.
    monkeypatch.setenv(COMPONENT_VARIABLE, "embedder")

    [(args, kwargs)] = serve_recording(monkeypatch, ["--serve", "reranker"])

    [factory] = args
    assert import_from_string(factory) is qp_model_server.model_server_from_environment
    assert (kwargs["factory"], kwargs["workers"]) == (True, 2)
    assert qp_model_server.os.environ[COMPONENT_VARIABLE] == "reranker"
