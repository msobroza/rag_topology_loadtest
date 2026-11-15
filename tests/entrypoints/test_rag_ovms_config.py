"""``rag-ovms-config``: what it prints, where its logs go, how it fails.

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
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.entrypoints.rag_ovms_config import main
from repo_paths import REPO_ROOT


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An export of both models, 2 CPUs, no RAG_* from the shell, no ``.env``."""
    for name in [n for n in os.environ if n.startswith(("RAG_", "DOMINO_"))]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("RAG_MODEL_CPUS", "2")
    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    write_ovms_export(models, "reranker", RagSettings().rerank_api_model)
    write_ovms_export(models, "embedder", RagSettings().embedding_api_model)
    return models


def stand_in_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behaviour: str = "answers"
) -> Path:
    record = tmp_path / "ovms.jsonl"
    bin_dir = write_ovms_stand_in(tmp_path / "bin", record, behaviour=behaviour)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return record


def args(part: str, models: Path, serving: Path) -> list[str]:
    return ["--serve", part, "--models-dir", str(models), "--serving-dir", str(serving)]


def test_stdout_is_the_config_path_and_nothing_else(
    models: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = stand_in_on_path(tmp_path, monkeypatch)
    serving = tmp_path / "serving"

    with caplog.at_level("INFO", logger="rag_load_test"):
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
    monkeypatch.setenv("RAG_OVMS_NUM_STREAMS", "1")
    monkeypatch.setenv("RAG_OVMS_THREADS", "2")
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
    assert line.startswith("rag-ovms-config: cannot write the reranker config: ")
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
    [line] = [x for x in captured.err.splitlines() if x.startswith("rag-ovms-config")]
    assert "exited with code 3" in line
    assert "Traceback" not in captured.err


def test_a_stream_count_above_the_threads_is_refused_naming_both(
    models: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RAG_OVMS_NUM_STREAMS", "4")
    monkeypatch.setenv("RAG_OVMS_THREADS", "2")

    assert main(args("reranker", models, tmp_path / "s")) == 1

    [line] = capsys.readouterr().err.splitlines()
    assert line.startswith("rag-ovms-config: cannot read the settings: ")
    assert "RAG_OVMS_NUM_STREAMS=4" in line
    assert "RAG_OVMS_THREADS=2" in line


def test_an_unknown_part_is_a_usage_error(models: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(args("llm", models, tmp_path / "s"))

    assert exit_info.value.code == 2


def test_the_command_is_a_project_script() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    scripts = pyproject["project"]["scripts"]
    assert (
        scripts["rag-ovms-config"] == "rag_load_test.entrypoints.rag_ovms_config:main"
    )
