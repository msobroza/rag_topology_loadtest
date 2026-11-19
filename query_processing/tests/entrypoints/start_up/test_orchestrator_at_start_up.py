"""``qp-serve`` names the orchestrator it starts on, runs each request on it, and an
orchestrator's refusal of the graph is a one-line usage error before anything opens.

The orchestrators that refuse or record are ones the test registers, chosen with
``QP_ORCHESTRATOR`` or the settings; the registry is left as it was found.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from query_processing.adapters.orchestrator.langgraph.langgraph_orchestrator import (
    LangGraphOrchestrator,
)
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints import qp_serve
from rag_components import FAKE_BACKEND_VARIABLES, shell_settings
from workflow_fakes import RefusingOrchestrator, register_orchestrator


def never_serve(app: object, **options: object) -> None:
    raise AssertionError(f"a server started with {options}: nothing was refused")


@pytest.fixture
def no_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """All-fake backends, no QP_* from the shell, an empty directory, no server run."""
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    for name, value in FAKE_BACKEND_VARIABLES.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(qp_serve.uvicorn, "run", never_serve)


@pytest.mark.usefixtures("no_server")
def test_an_orchestrators_refusal_is_one_line_before_anything_opens(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    register_orchestrator(monkeypatch, "refusing", RefusingOrchestrator)
    monkeypatch.setenv("QP_ORCHESTRATOR", "refusing")

    with pytest.raises(SystemExit) as exited:
        qp_serve.main([])

    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: qp-serve")
    assert "Traceback" not in err
    [line] = [line for line in err.splitlines() if line.startswith("qp-serve: ")]
    assert line == (
        "qp-serve: error: orchestrator 'refusing': stage 'rerank' has a "
        "conditional edge, which this library does not run (expected plain "
        "edges only); the default orchestrator 'langgraph' accepts this graph"
    )


def test_the_start_up_log_line_names_the_orchestrator(
    fake_settings: WorkflowSettings, caplog: pytest.LogCaptureFixture
) -> None:
    app = qp_serve.create_app(CompositionRoot(fake_settings))

    with caplog.at_level("INFO", logger="query_processing"), TestClient(app):
        pass

    [logged] = [
        json.loads(r.getMessage())
        for r in caplog.records
        if '"event": "rag_startup"' in r.getMessage()
    ]
    assert logged["orchestrator"] == "langgraph"


class RecordingOrchestrator(LangGraphOrchestrator):
    """LangGraph's orchestrator, which records the question of each request it runs."""

    questions: ClassVar[list[str]] = []

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        type(self).questions.append(state["conversational_query"])
        return await super().ainvoke(state)


def test_a_request_runs_on_the_orchestrator_the_settings_name(
    fake_settings: WorkflowSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_orchestrator(monkeypatch, "recording", RecordingOrchestrator)
    monkeypatch.setattr(RecordingOrchestrator, "questions", [])
    settings = fake_settings.model_copy(update={"orchestrator": "recording"})

    with TestClient(qp_serve.create_app(CompositionRoot(settings))) as client:
        answered = client.post("/query", json={"question": "How many days off?"})

    assert answered.status_code == 200
    assert RecordingOrchestrator.questions == ["How many days off?"]
