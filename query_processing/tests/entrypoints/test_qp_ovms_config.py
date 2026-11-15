"""``qp-ovms-config``: what it prints, where its logs go, how it fails.

A stand-in ``ovms`` is first on PATH; the search itself is tested in
``tests/composition/test_ovms_calibration.py``.
"""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import pytest

from ovms_stand_ins import ovms_starts, write_ovms_export, write_ovms_stand_in
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.entrypoints.qp_ovms_config import main
from rag_components import shell_settings
from rag_graphs import FIXTURE_WORKFLOW
from repo_paths import TOOLKIT_ROOT


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An export of both models, 2 CPUs, no QP_* from the shell, no ``.env``."""
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    monkeypatch.setenv("QP_LOCAL_MODELS__CPUS", "2")
    monkeypatch.chdir(tmp_path)
    components = WorkflowSettings(_env_file=None).components  # type: ignore[call-arg]
    models = tmp_path / "models"
    write_ovms_export(models, "reranker", components["reranker"].alias)
    write_ovms_export(models, "embedder", components["embedder"].alias)
    return models


def stand_in_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behaviour: str = "answers"
) -> Path:
    record = tmp_path / "ovms.jsonl"
    bin_dir = write_ovms_stand_in(tmp_path / "bin", record, behaviour=behaviour)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return record


def args(component: str, models: Path, serving: Path) -> list[str]:
    return [
        "--serve",
        component,
        "--models-dir",
        str(models),
        "--serving-dir",
        str(serving),
    ]


def test_stdout_is_the_config_path_and_nothing_else(
    models: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = stand_in_on_path(tmp_path, monkeypatch)
    serving = tmp_path / "serving"

    with caplog.at_level("INFO", logger="query_processing"):
        assert main(args("reranker", models, serving)) == 0

    assert capsys.readouterr().out == f"{serving / 'config_reranker.json'}\n"
    assert len(ovms_starts(record)) >= 2  # calibrated: the settings leave both at 0
    events = [json.loads(r.message)["event"] for r in caplog.records]
    assert {"stage_calibrated", "ovms_serving_config"} <= set(events)


def test_both_pinned_start_no_ovms(
    models: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("QP_OVMS_NUM_STREAMS", "1")
    monkeypatch.setenv("QP_OVMS_THREADS", "2")
    monkeypatch.setenv("PATH", str(tmp_path / "no-ovms-here"))

    assert main(args("embedder", models, tmp_path / "s")) == 0

    assert capsys.readouterr().out.strip().endswith("config_embedder.json")


def test_a_missing_export_is_one_stderr_line(
    tmp_path: Path, models: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args("reranker", tmp_path / "nowhere", tmp_path / "s")) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert line.startswith("qp-ovms-config: cannot write the reranker config: ")
    assert "ovms/export_models.sh" in line


def test_an_ovms_that_exits_is_one_stderr_line_with_its_code(
    models: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stand_in_on_path(tmp_path, monkeypatch, behaviour="exits")

    assert main(args("reranker", models, tmp_path / "s")) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = [x for x in captured.err.splitlines() if x.startswith("qp-ovms-config")]
    assert "exited with code 3" in line
    assert "Traceback" not in captured.err


def test_a_stream_count_above_the_threads_is_refused_naming_both(
    models: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("QP_OVMS_NUM_STREAMS", "4")
    monkeypatch.setenv("QP_OVMS_THREADS", "2")

    assert main(args("reranker", models, tmp_path / "s")) == 1

    [line] = capsys.readouterr().err.splitlines()
    assert line.startswith("qp-ovms-config: cannot read the settings: ")
    assert "QP_OVMS_NUM_STREAMS=4" in line
    assert "QP_OVMS_THREADS=2" in line


def int8_in_the_workflow_file_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workflow file that states int8, no variable: both counts pinned, no ovms."""
    workflow = tmp_path / "int8-workflow.yaml"
    stated = "\nlocal_models:\n  openvino_weight_format: int8\n"
    workflow.write_text(FIXTURE_WORKFLOW.read_text() + stated)
    monkeypatch.setenv("QP_WORKFLOW_FILE", str(workflow))
    monkeypatch.setenv("QP_OVMS_NUM_STREAMS", "1")
    monkeypatch.setenv("QP_OVMS_THREADS", "1")


def test_a_weight_format_written_only_in_the_workflow_file_is_the_one_asked(
    models: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The launch script sees the variable only; this command reads the settings,
    # so an int8 export is served when the workflow file alone asks for int8.
    int8_in_the_workflow_file_only(tmp_path, monkeypatch)
    (models / "weight-format").write_text("int8\n")

    assert main(args("reranker", models, tmp_path / "s")) == 0

    assert capsys.readouterr().out.strip().endswith("config_reranker.json")


def test_an_export_of_another_weight_format_than_the_settings_ask_is_one_stderr_line(
    models: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    int8_in_the_workflow_file_only(tmp_path, monkeypatch)  # the folder holds fp32

    assert main(args("reranker", models, tmp_path / "s")) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert line.startswith("qp-ovms-config: cannot write the reranker config: ")
    assert "weight format fp32, but int8 was asked" in line
    assert "QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT=fp32" in line


@pytest.mark.parametrize("name", ["llm", "no-such-component"])
def test_an_unknown_component_is_a_usage_error(
    models: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    # Only an embedder or a reranker of the workflow file can be served by OVMS.
    with pytest.raises(SystemExit) as exit_info:
        main(args(name, models, tmp_path / "s"))

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "expected an embedder or a reranker" in captured.err
    assert "embedder, reranker" in captured.err


def test_the_command_is_a_project_script() -> None:
    pyproject = tomllib.loads((TOOLKIT_ROOT / "pyproject.toml").read_text())

    scripts = pyproject["project"]["scripts"]
    assert (
        scripts["qp-ovms-config"] == "query_processing.entrypoints.qp_ovms_config:main"
    )
