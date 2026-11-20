"""A component a use case defines answers a stage of the workflow app.

The workflow file names the test extension under ``extensions``; its
``shouter`` component type and its implementations register on their class
line, and a plain call to the shouter writes the answer ``/query`` returns.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints import qp_serve
from query_processing.entrypoints.qp_serve import create_app
from rag_components import FAKE_BACKEND_VARIABLES, FAKE_BACKENDS, shell_settings
from shout_components import SHOUT_EXTENSION, write_shout_workflow

QUESTION = "How many paid vacation days?"
SettingsFactory = Callable[..., WorkflowSettings]


@pytest.fixture
def shout_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A workflow file whose last stage calls a shouter, named in QP_WORKFLOW_FILE."""
    path = write_shout_workflow(
        tmp_path / "workflow.yaml",
        [SHOUT_EXTENSION],
        {"shouter": {"backend": "local", "local": {"suffix": "!"}}},
    )
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(path))
    return path


def ask(settings: WorkflowSettings, path: str = "/query") -> dict[str, Any]:
    with TestClient(create_app(CompositionRoot(settings))) as app:
        response = app.post(path, json={"question": QUESTION})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.usefixtures("shout_workflow")
def test_a_stage_that_calls_a_use_cases_component_answers_through_the_app(
    settings_factory: SettingsFactory,
) -> None:
    body = ask(settings_factory(**FAKE_BACKENDS))

    assert body["answer"] == QUESTION.upper() + "!"
    assert len(body["records"]) == 2
    assert body["timings_ms"]["shout"] > 0


@pytest.mark.usefixtures("shout_workflow")
def test_the_backend_a_variable_names_picks_the_implementation(
    settings_factory: SettingsFactory,
) -> None:
    body = ask(settings_factory(**FAKE_BACKENDS, components__shouter__backend="fake"))

    assert body["answer"] == QUESTION


def test_a_backend_the_type_does_not_have_stops_the_app_before_it_starts(
    shout_workflow: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in shell_settings(os.environ):
        monkeypatch.delenv(name)
    monkeypatch.chdir(shout_workflow.parent)  # no .env of the developer's
    for name, value in FAKE_BACKEND_VARIABLES.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("QP_COMPONENTS__SHOUTER__BACKEND", "loud")
    monkeypatch.setattr(qp_serve.uvicorn, "run", never_serve)

    with pytest.raises(SystemExit) as stopped:
        qp_serve.main([])

    assert stopped.value.code == 2
    assert (
        "unknown shouter backend 'loud' (expected one of: 'fake', 'local', 'served')"
        in (capsys.readouterr().err)
    )


def never_serve(app: object, **options: object) -> None:
    raise AssertionError(f"a server started with {options}: nothing was refused")
