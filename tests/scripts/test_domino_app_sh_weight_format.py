"""domino/app.sh serves an OVMS export only at the weight format asked.

An OVMS models folder holds one export, of one weight format, recorded in its
``weight-format`` file by ovms/export_models.sh; a folder without the file was
exported before the record existed, always at int8. The stand-ins of
tests/helpers/launch_script_stand_ins.py record what would have run.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from launch_script_stand_ins import (
    APP_SH,
    log,
    make_stand_ins,
    models_dir,
    stand_in_env,
)
from rag_load_test.config.rag_settings import RagSettings


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    return make_stand_ins(tmp_path)


def run_ovms_reranker(
    tmp_path: Path, bin_dir: Path, models: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(APP_SH)],
        env=stand_in_env(
            tmp_path,
            bin_dir,
            RAG_ROLE="ovms-reranker",
            RAG_OVMS_MODELS_DIR=str(models),
            **env,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_app_sh_repeats_the_settings_default_weight_format() -> None:
    # app.sh cannot ask RagSettings (the OVMS roles install no project), so it
    # repeats the default; this keeps the two equal.
    [bash_default] = re.findall(
        r"\$\{RAG_OPENVINO_WEIGHT_FORMAT:-([a-z0-9]+)\}", APP_SH.read_text()
    )

    assert bash_default == RagSettings.model_fields["openvino_weight_format"].default


@pytest.mark.parametrize(
    ("recorded", "asked"), [("fp32", None), ("int8", "int8"), ("fp32", "fp32")]
)
def test_an_export_of_the_weight_format_asked_is_served(
    tmp_path: Path, bin_dir: Path, recorded: str, asked: str | None
) -> None:
    env = {"RAG_OPENVINO_WEIGHT_FORMAT": asked} if asked else {}
    models = models_dir(tmp_path, recorded)

    result = run_ovms_reranker(tmp_path, bin_dir, models, **env)

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "ovms") == (
        f"--rest_port 8888 --config_path {models}/config_reranker.json\n"
    )


def test_a_folder_without_a_record_is_served_as_int8(
    tmp_path: Path, bin_dir: Path
) -> None:
    models = models_dir(tmp_path)
    (models / "weight-format").unlink()

    result = run_ovms_reranker(
        tmp_path, bin_dir, models, RAG_OPENVINO_WEIGHT_FORMAT="int8"
    )

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "ovms") != ""


@pytest.mark.parametrize(
    ("recorded", "asked", "env"),
    [
        ("int8", "fp32", {}),  # unset: fp32
        ("fp32", "int8", {"RAG_OPENVINO_WEIGHT_FORMAT": "int8"}),
        (None, "fp32", {}),  # exported before the record: int8
    ],
    ids=["int8-folder-default", "fp32-folder-int8-asked", "folder-without-record"],
)
def test_another_weight_format_stops_the_app_before_ovms_starts(
    tmp_path: Path,
    bin_dir: Path,
    recorded: str | None,
    asked: str,
    env: dict[str, str],
) -> None:
    models = models_dir(tmp_path, recorded or "unused")
    found = recorded or "int8"
    if recorded is None:
        (models / "weight-format").unlink()

    result = run_ovms_reranker(tmp_path, bin_dir, models, **env)

    assert result.returncode == 1
    assert log(tmp_path, "ovms") == ""
    message = result.stderr
    assert message.startswith("app.sh[ovms-reranker]: ")
    assert str(models) in message
    assert f"weight format {found}, but {asked} was asked" in message
    assert f"set RAG_OPENVINO_WEIGHT_FORMAT={found}" in message
    assert "point RAG_OVMS_MODELS_DIR at another folder" in message
