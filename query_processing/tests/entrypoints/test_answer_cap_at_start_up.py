"""A generate stage's answer cap below 1 is refused when ``qp-serve`` starts.

The cases of ``test_wrong_graph_at_start_up.py`` in one more: a usage error that
names the stage, the value found and the shape expected, before anything opens.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from query_processing.config.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.entrypoints import qp_serve
from rag_components import FAKE_BACKEND_VARIABLES, shell_settings
from rag_graphs import fixture_graph, write_workflow_file


@pytest.mark.parametrize("cap", [0, -1])
def test_a_cap_below_one_stops_the_start_up_naming_the_stage(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    cap: int,
) -> None:
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    for name, value in FAKE_BACKEND_VARIABLES.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    served: list[object] = []
    monkeypatch.setattr(qp_serve.uvicorn, "run", lambda app, **_: served.append(app))
    graph = fixture_graph()
    graph["stages"]["generate"]["max_tokens"] = cap
    workflow = write_workflow_file(tmp_path / "workflow.yaml", graph)
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))

    with pytest.raises(SystemExit) as exited:
        qp_serve.main([])

    assert (exited.value.code, served) == (2, [])
    err = capsys.readouterr().err
    assert "stage 'generate'" in err
    assert (
        f"max_tokens is {cap} (expected a whole number of at least 1, or no "
        "max_tokens for no cap)"
    ) in err
    assert "Traceback" not in err
