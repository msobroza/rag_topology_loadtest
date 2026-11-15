"""``rag-calibrate``: which components it calibrates, what it prints, how it fails.

The models are ``FakeSentenceTransformersModule``'s, on the real clock; the
search itself is tested in ``tests/composition/test_composition_root_calibration.py``.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tomllib
from pathlib import Path

import pytest
from dotenv import dotenv_values

from ovms_stand_ins import ovms_starts, write_ovms_export, write_ovms_stand_in
from rag_load_test.composition.calibrated_stages import ComponentCalibration
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.calibration import CalibrationReport, StageTuning
from rag_load_test.entrypoints.rag_calibrate import env_lines, main
from rag_load_test.infrastructure.local_models.local_model import exported_model_dir
from repo_paths import REPO_ROOT
from sentence_transformers_fakes import (
    FakeSentenceTransformersModule,
    fake_openvino_modules,
)

RERANK_KEYS = ["RAG_RERANK_THREADS", "RAG_RERANK_BATCH_SIZE"]
EMBED_KEYS = ["RAG_EMBEDDING_THREADS"]
OVMS_KEYS = ["RAG_OVMS_NUM_STREAMS", "RAG_OVMS_THREADS"]


@pytest.fixture
def fake_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> FakeSentenceTransformersModule:
    """Fake models, 2 CPUs, one worker, torch; no RAG_* from the shell, no ``.env``."""
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    for name, fake in fake_openvino_modules().items():
        monkeypatch.setitem(sys.modules, name, fake)
    for name in [n for n in os.environ if n.startswith(("RAG_", "DOMINO_"))]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("RAG_LOCAL_MODEL_RUNTIME", "torch")
    monkeypatch.setenv("RAG_MODEL_CPUS", "2")
    monkeypatch.setenv("RAG_MODEL_WORKERS", "1")  # so each share is 2 threads
    monkeypatch.chdir(tmp_path)
    return module


def stdout_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().out.splitlines()


def test_both_components_by_default(
    fake_weights: FakeSentenceTransformersModule, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([]) == 0

    lines = stdout_lines(capsys)
    comments = [line for line in lines if line.startswith("#")]
    assert [comment.split(":")[0] for comment in comments] == ["# embed", "# rerank"]
    assert [line.split("=")[0] for line in lines if "=" in line and line[0] != "#"] == [
        *EMBED_KEYS,
        *RERANK_KEYS,
    ]


@pytest.mark.parametrize(
    ("part", "keys"), [("embedder", EMBED_KEYS), ("reranker", RERANK_KEYS)]
)
def test_only_calibrates_one_component(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    part: str,
    keys: list[str],
) -> None:
    assert main(["--only", part]) == 0

    out = capsys.readouterr().out
    assert list(dotenv_values(stream=io.StringIO(out))) == keys
    assert {loaded.split(":")[0] for loaded in fake_weights.loaded} == {part}


def test_stdout_is_comments_and_assignments_that_read_back_as_an_env_file(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    assert main(["--only", "reranker"]) == 0

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("# rerank: timed on 2 CPUs under torch, ")
    assert " s per request; tried 6 candidates in " in lines[0]
    assert all(line.startswith("#") or _assignment(line) for line in lines)
    env_file = tmp_path / ".env"
    env_file.write_text(out)
    pinned = dotenv_values(env_file)
    assert list(pinned) == RERANK_KEYS
    assert all(value is not None and int(value) >= 1 for value in pinned.values())


def test_the_comment_says_when_the_budget_cut_the_search_short(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CALIBRATION_BUDGET_S", "1e-9")

    assert main(["--only", "reranker"]) == 0

    [comment, *assignments] = stdout_lines(capsys)
    assert "; tried 1 candidate in " in comment
    assert comment.endswith(", cut short by RAG_CALIBRATION_BUDGET_S=1e-09")
    assert assignments == ["RAG_RERANK_THREADS=2", "RAG_RERANK_BATCH_SIZE=20"]


def test_the_comment_says_when_the_first_candidate_was_kept_untimed() -> None:
    # A served stage whose first trial outlived the budget (GCP sitting of
    # 2026-11-15, 16 requests on one OVMS stream): no trial, cut short.
    report = CalibrationReport(
        stage="rerank",
        chosen=StageTuning(8, 20, streams=1),
        trials=(),
        seconds=95.2,
        budget_s=90.0,
        cut_short=True,
        requests=16,
    )
    pinned = {"RAG_OVMS_NUM_STREAMS": 1, "RAG_OVMS_THREADS": 8}

    [comment, *_] = env_lines(ComponentCalibration(report, 8.0, "ovms", pinned))

    assert comment == (
        "# rerank: timed on 8 CPUs under ovms, the first candidate, not timed; "
        "tried 0 candidates in 95.2 s, cut short by RAG_CALIBRATION_BUDGET_S=90"
    )


def test_the_logs_stay_off_stdout(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="rag_load_test"):
        assert main(["--only", "embedder"]) == 0

    events = [json.loads(record.message)["event"] for record in caplog.records]
    assert "stage_calibrated" in events
    out = capsys.readouterr().out
    assert "stage_calibrated" not in out
    assert "{" not in out


def test_a_bad_setting_is_one_stderr_line_naming_the_value(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CALIBRATION_ROUNDS", "-4")

    assert main([]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert line.startswith("rag-calibrate: cannot read the settings: ")
    assert "calibration_rounds" in line
    assert "-4" in line


def test_a_model_that_cannot_be_provisioned_is_one_stderr_line(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_weights.save_error = OSError("No space left on device")
    monkeypatch.setenv("RAG_LOCAL_MODEL_RUNTIME", "openvino")
    monkeypatch.setenv("RAG_OPENVINO_MODELS_DIR", str(tmp_path / "models"))

    assert main(["--only", "reranker"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert line.startswith("rag-calibrate: cannot calibrate reranker: ")
    assert "No space left on device" in line
    assert "Traceback" not in captured.err


def test_a_missing_openvino_package_is_one_stderr_line_when_the_folder_exists(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Regression, sitting of 15 November 2026: an exported folder skipped the
    # package check and the library's bare Exception printed a traceback. None in
    # sys.modules makes the import fail as an absent package does.
    monkeypatch.setitem(sys.modules, "optimum.intel.openvino", None)
    models = tmp_path / "models"
    monkeypatch.setenv("RAG_LOCAL_MODEL_RUNTIME", "openvino")
    monkeypatch.setenv("RAG_OPENVINO_MODELS_DIR", str(models))
    rerank_model = RagSettings(_env_file=None).rerank_model  # type: ignore[call-arg]
    exported_model_dir(str(models), rerank_model, weight_format="fp32").mkdir(
        parents=True
    )

    assert main(["--only", "reranker"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert line.startswith("rag-calibrate: cannot calibrate reranker: ")
    assert "'optimum.intel.openvino'" in line
    assert "sentence-transformers[openvino]" in line
    assert "Traceback" not in captured.err
    assert fake_weights.loaded == []


def test_an_unknown_only_value_is_a_usage_error(
    fake_weights: FakeSentenceTransformersModule,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--only", "llm"])

    assert exit_info.value.code == 2
    assert fake_weights.loaded == []


def test_ovms_models_calibrates_ovms_instead_of_the_in_process_model(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = tmp_path / "ovms-models"
    write_ovms_export(models, "reranker", RagSettings().rerank_api_model)
    record = tmp_path / "ovms.jsonl"
    bin_dir = write_ovms_stand_in(tmp_path / "bin", record)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    assert main(["--ovms-models", str(models), "--only", "reranker"]) == 0

    [comment, *assignments] = stdout_lines(capsys)
    assert comment.startswith("# rerank: timed on 2 CPUs under ovms, ")
    assert " s for 4 requests at once; tried " in comment
    assert [line.split("=")[0] for line in assignments] == OVMS_KEYS
    assert fake_weights.loaded == []  # no in-process model loaded
    assert len(ovms_starts(record)) >= 2


def test_ovms_models_prints_the_same_two_variables_for_the_embedder(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models = tmp_path / "ovms-models"
    write_ovms_export(models, "embedder", RagSettings().embedding_api_model)
    bin_dir = write_ovms_stand_in(tmp_path / "bin", tmp_path / "ovms.jsonl")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    assert main(["--ovms-models", str(models), "--only", "embedder"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("# embed: timed on 2 CPUs under ovms, ")
    assert list(dotenv_values(stream=io.StringIO(out))) == OVMS_KEYS


def test_ovms_models_without_an_export_is_one_stderr_line(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    assert main(["--ovms-models", str(tmp_path / "none"), "--only", "reranker"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert line.startswith("rag-calibrate: cannot calibrate reranker: ")
    assert "ovms/export_models.sh" in line


def test_the_command_is_a_project_script() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    scripts = pyproject["project"]["scripts"]
    assert scripts["rag-calibrate"] == "rag_load_test.entrypoints.rag_calibrate:main"


def _assignment(line: str) -> bool:
    name, _, value = line.partition("=")
    return name.isupper() and value.isdigit()
