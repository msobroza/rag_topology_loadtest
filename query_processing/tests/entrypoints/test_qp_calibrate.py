"""``qp-calibrate``: which components it calibrates, what it prints, how it fails.

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
import yaml
from dotenv import dotenv_values

from query_processing.deploy.deploy_settings import DeploySettings
from query_processing.deploy.overrides import override_exports
from query_processing.entrypoints.qp_calibrate import main
from rag_components import shell_settings
from repo_paths import TOOLKIT_ROOT
from sentence_transformers_fakes import (
    FakeSentenceTransformersModule,
    fake_openvino_modules,
)

RERANK_KEYS = [
    "QP_COMPONENTS__RERANKER__LOCAL__THREADS",
    "QP_COMPONENTS__RERANKER__LOCAL__BATCH_SIZE",
]
EMBED_KEYS = ["QP_COMPONENTS__EMBEDDER__LOCAL__THREADS"]
# Where the second comment of each component says to pin it in deploy.yaml.
OVERRIDES_HINT = "deploy.yaml, under overrides.components"


@pytest.fixture
def fake_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> FakeSentenceTransformersModule:
    """Fake models, 2 CPUs, one worker, torch; no QP_* from the shell, no ``.env``."""
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    for name, fake in fake_openvino_modules().items():
        monkeypatch.setitem(sys.modules, name, fake)
    for name in shell_settings(os.environ, ("QP_", "DOMINO_")):
        monkeypatch.delenv(name)
    monkeypatch.setenv("QP_LOCAL_MODELS__RUNTIME", "torch")
    monkeypatch.setenv("QP_LOCAL_MODELS__CPUS", "2")
    monkeypatch.setenv("QP_LOCAL_MODELS__WORKERS", "1")  # so each share is 2 threads
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
    assert [comment.split(":")[0] for comment in comments] == [
        "# embedder (embed)",
        f"# {OVERRIDES_HINT}",
        "# reranker (rerank)",
        f"# {OVERRIDES_HINT}",
    ]
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


def test_a_component_of_any_name_is_pinned_by_its_environment_paths(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A component declared by its environment paths alone, beside the file's.
    monkeypatch.setenv("QP_COMPONENTS__COARSE__TYPE", "reranker")
    monkeypatch.setenv("QP_COMPONENTS__COARSE__BACKEND", "local")
    monkeypatch.setenv("QP_COMPONENTS__COARSE__SOURCE_ID", "org/coarse")

    assert main(["--only", "coarse"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("# coarse (rerank): ")
    assert list(dotenv_values(stream=io.StringIO(out))) == [
        "QP_COMPONENTS__COARSE__LOCAL__THREADS",
        "QP_COMPONENTS__COARSE__LOCAL__BATCH_SIZE",
    ]
    assert set(fake_weights.loaded) == {"reranker:org/coarse"}  # once per thread count


def test_stdout_is_comments_and_assignments_that_read_back_as_an_env_file(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    assert main(["--only", "reranker"]) == 0

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("# reranker (rerank): timed on 2 CPUs under torch, ")
    assert " s per request; tried 6 candidates in " in lines[0]
    assert all(line.startswith("#") or _assignment(line) for line in lines)
    env_file = tmp_path / ".env"
    env_file.write_text(out)
    pinned = dotenv_values(env_file)
    assert list(pinned) == RERANK_KEYS
    assert all(value is not None and int(value) >= 1 for value in pinned.values())


def test_what_it_prints_pins_the_same_values_under_the_overrides(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0

    out = capsys.readouterr().out
    hints = [
        line.removeprefix(f"# {OVERRIDES_HINT}: ")
        for line in out.splitlines()
        if line.startswith(f"# {OVERRIDES_HINT}: ")
    ]
    # Each line is one entry of overrides.components in deploy.yaml, and the
    # deploy tool exports it under the variables the .env lines name.
    components = {k: v for hint in hints for k, v in yaml.safe_load(hint).items()}
    assert list(components) == ["embedder", "reranker"]
    settings = DeploySettings(
        app_name_prefix="rag", overrides={"components": components}
    )
    exports = override_exports(settings)
    assert exports == dotenv_values(stream=io.StringIO(out))


def test_the_comment_says_when_the_budget_cut_the_search_short(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_CALIBRATION__BUDGET_S", "1e-9")

    assert main(["--only", "reranker"]) == 0

    [comment, _, *assignments] = stdout_lines(capsys)
    assert "; tried 1 candidate in " in comment
    assert comment.endswith(", cut short by QP_CALIBRATION__BUDGET_S=1e-09")
    assert assignments == [
        "QP_COMPONENTS__RERANKER__LOCAL__THREADS=2",
        "QP_COMPONENTS__RERANKER__LOCAL__BATCH_SIZE=20",
    ]


def test_the_logs_stay_off_stdout(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="query_processing"):
        assert main(["--only", "embedder"]) == 0

    events = [json.loads(record.message)["event"] for record in caplog.records]
    assert "stage_calibrated" in events
    out = capsys.readouterr().out
    assert "stage_calibrated" not in out
    assert not [line for line in out.splitlines() if line.startswith("{")]


def test_a_bad_setting_is_one_stderr_line_naming_the_value(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QP_CALIBRATION__ROUNDS", "-4")

    assert main([]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert line.startswith("qp-calibrate: cannot read the settings: ")
    assert "calibration.rounds" in line
    assert "-4" in line


def test_a_model_that_cannot_be_provisioned_is_one_stderr_line(
    fake_weights: FakeSentenceTransformersModule,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_weights.save_error = OSError("No space left on device")
    monkeypatch.setenv("QP_LOCAL_MODELS__RUNTIME", "openvino")
    monkeypatch.setenv("QP_LOCAL_MODELS__OPENVINO_DIR", str(tmp_path / "models"))

    assert main(["--only", "reranker"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    [line] = captured.err.splitlines()
    assert line.startswith("qp-calibrate: cannot calibrate reranker: ")
    assert "No space left on device" in line
    assert "Traceback" not in captured.err


def test_an_unknown_only_value_is_a_usage_error(
    fake_weights: FakeSentenceTransformersModule,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--only", "llm"])

    assert exit_info.value.code == 2
    assert fake_weights.loaded == []


def test_the_command_is_a_project_script() -> None:
    pyproject = tomllib.loads((TOOLKIT_ROOT / "pyproject.toml").read_text())

    scripts = pyproject["project"]["scripts"]
    assert scripts["qp-calibrate"] == "query_processing.entrypoints.qp_calibrate:main"


def _assignment(line: str) -> bool:
    name, _, value = line.partition("=")
    return name.isupper() and value.isdigit()
