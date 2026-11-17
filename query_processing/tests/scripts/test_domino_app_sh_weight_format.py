"""An OVMS model server serves an OVMS export only at the weight format asked.

An OVMS models folder holds one export, of one weight format, recorded in its
``weight-format`` file by ovms/export_models.sh; a folder without the file was
exported before the record existed, always at int8. The launch script checks the
folder itself when the variable says what is asked. Unset, what is asked is the
settings' to say (the workflow file may state it), so the check is left to
``qp-ovms-config``, which reads them (tests/entrypoints/test_qp_ovms_config.py);
without that command the script checks against the default
(test_domino_app_sh_ovms_config.py). The stand-ins of
tests/helpers/launch_script_stand_ins.py record what would have run.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from launch_script_stand_ins import (
    APP_SH,
    log,
    make_stand_ins,
    model_server_env,
    models_dir,
    served_config,
    stand_in_env,
)
from ovms_stand_ins import write_ovms_export
from query_processing.config.shared_settings import LocalModelsSettings
from rag_graphs import FIXTURE_WORKFLOW
from repo_paths import TOOLKIT_ROOT

OVMS_SH = TOOLKIT_ROOT / "domino" / "model_servers" / "ovms.sh"


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
            **model_server_env("ovms", "reranker"),
            QP_OVMS_MODELS_DIR=str(models),
            **env,
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_ovms_sh_repeats_the_settings_default_weight_format() -> None:
    # Without qp-ovms-config ovms.sh cannot ask LocalModelsSettings, so it
    # repeats the default; this keeps the two equal.
    [bash_default] = re.findall(
        r"\$\{QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT:-([a-z0-9]+)\}",
        OVMS_SH.read_text(),
    )

    python_default = LocalModelsSettings.model_fields["openvino_weight_format"]
    assert bash_default == python_default.default


@pytest.mark.parametrize(
    ("recorded", "asked"), [("fp32", None), ("int8", "int8"), ("fp32", "fp32")]
)
def test_an_export_of_the_weight_format_asked_is_served(
    tmp_path: Path, bin_dir: Path, recorded: str, asked: str | None
) -> None:
    env = {"QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT": asked} if asked else {}
    models = models_dir(tmp_path, recorded)

    result = run_ovms_reranker(tmp_path, bin_dir, models, **env)

    assert result.returncode == 0, result.stderr
    # OVMS starts on the config qp-ovms-config printed for this export.
    assert f"--models-dir {models} " in log(tmp_path, "qp-ovms-config")
    assert log(tmp_path, "ovms") == (
        f"--rest_port 8888 --config_path {served_config(tmp_path)}\n"
    )


def test_a_folder_without_a_record_is_served_as_int8(
    tmp_path: Path, bin_dir: Path
) -> None:
    models = models_dir(tmp_path)
    (models / "weight-format").unlink()

    result = run_ovms_reranker(
        tmp_path, bin_dir, models, QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT="int8"
    )

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "ovms") != ""


@pytest.mark.parametrize("recorded", ["int8", None], ids=["int8", "no-record"])
def test_an_unset_variable_leaves_the_check_to_qp_ovms_config(
    tmp_path: Path, bin_dir: Path, recorded: str | None
) -> None:
    # Unset, the settings say what is asked, and the workflow file may state
    # int8: the script must not refuse an int8 folder on its own default.
    models = models_dir(tmp_path, recorded or "unused")
    if recorded is None:
        (models / "weight-format").unlink()

    result = run_ovms_reranker(tmp_path, bin_dir, models)

    assert result.returncode == 0, result.stderr
    assert f"--models-dir {models} " in log(tmp_path, "qp-ovms-config")
    assert log(tmp_path, "ovms") == (
        f"--rest_port 8888 --config_path {served_config(tmp_path)}\n"
    )


@pytest.mark.parametrize(
    ("recorded", "asked"),
    [("int8", "fp32"), ("fp32", "int8"), (None, "fp32")],
    ids=["int8-folder-fp32-asked", "fp32-folder-int8-asked", "folder-without-record"],
)
def test_another_weight_format_than_the_variable_asks_stops_the_app_at_once(
    tmp_path: Path, bin_dir: Path, recorded: str | None, asked: str
) -> None:
    models = models_dir(tmp_path, recorded or "unused")
    found = recorded or "int8"  # exported before the record: int8
    if recorded is None:
        (models / "weight-format").unlink()
    env = {"QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT": asked}

    result = run_ovms_reranker(tmp_path, bin_dir, models, **env)

    assert result.returncode == 1
    assert log(tmp_path, "ovms") == ""
    assert log(tmp_path, "qp-ovms-config") == ""  # refused before it is asked
    message = result.stderr
    assert message.startswith("app.sh[model-server ovms reranker]: ")
    assert str(models) in message
    assert f"weight format {found}, but {asked} was asked" in message
    assert f"set QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT={found}" in message
    assert "point QP_OVMS_MODELS_DIR at another folder" in message


# --- the real qp-ovms-config: what reads the workflow file -------------------------


def put_the_real_qp_ovms_config_on_path(bin_dir: Path) -> None:
    """Replace the stand-in by the real command, run by this interpreter."""
    run_main = (
        "from query_processing.entrypoints.qp_ovms_config import main; "
        "raise SystemExit(main())"
    )
    script = bin_dir / "qp-ovms-config"
    script.write_text(
        f"#!/usr/bin/env bash\nexec '{sys.executable}' -c '{run_main}' \"$@\"\n"
    )
    script.chmod(0o755)


@pytest.mark.parametrize("recorded", ["int8", "fp32"])
def test_a_weight_format_stated_only_in_the_workflow_file_is_the_one_checked(
    tmp_path: Path, bin_dir: Path, recorded: str
) -> None:
    # The use case states int8 in its workflow file and sets no variable. The
    # script cannot see that, so it must leave the check to qp-ovms-config: an
    # int8 export is served, an fp32 one refused. Both counts are pinned, so the
    # real command starts no ovms of its own.
    put_the_real_qp_ovms_config_on_path(bin_dir)
    workflow = tmp_path / "workflow.yaml"
    stated = "\nlocal_models:\n  openvino_weight_format: int8\n"
    workflow.write_text(FIXTURE_WORKFLOW.read_text() + stated)
    models = tmp_path / "models"
    write_ovms_export(models, "reranker", "fixture-reranker", recorded)

    result = run_ovms_reranker(
        tmp_path,
        bin_dir,
        models,
        QP_WORKFLOW_FILE=str(workflow),
        QP_OVMS_NUM_STREAMS="1",
        QP_OVMS_THREADS="1",
        PYTHONPATH=str(TOOLKIT_ROOT / "src"),
    )

    if recorded == "int8":
        assert result.returncode == 0, result.stderr
        [started] = log(tmp_path, "ovms").splitlines()
        assert started.startswith("--rest_port 8888 --config_path ")
        assert started.endswith("/config_reranker.json")
        assert str(models) not in started  # the App's own serving folder
    else:
        assert result.returncode == 1
        assert log(tmp_path, "ovms") == ""
        assert "weight format fp32, but int8 was asked" in result.stderr
        assert "qp-ovms-config: cannot write the reranker config: " in result.stderr
