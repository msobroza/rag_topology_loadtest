"""domino/app.sh as an OVMS App: the server flags ``qp-ovms-config`` prints reach ovms.

The real ``qp-ovms-config`` runs here, so the command line is the one the
settings give: at the defaults it is today's, word for word. Both counts are
pinned on a reranker's App so the command starts no ovms of its own. The other
binaries are the stand-ins of tests/helpers/launch_script_stand_ins.py, which
record what would have run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from launch_script_stand_ins import (
    APP_SH,
    log,
    make_stand_ins,
    model_server_env,
    put_the_real_toolkit_command_on_path,
    stand_in_env,
    write_fake,
)
from ovms_stand_ins import write_ovms_export
from repo_paths import TOOLKIT_ROOT

LLM_SOURCE_ID = "OpenVINO/Fixture-Chat-int4-ov"
SERVED_LLM = f"llm llm {LLM_SOURCE_ID} fixture-chat fp32"
# Pinned, so the real qp-ovms-config writes its config without starting ovms.
PINNED = {"QP_OVMS_NUM_STREAMS": "1", "QP_OVMS_THREADS": "1"}


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """The stand-ins, with the toolkit's own ``qp-ovms-config``."""
    bin_dir = make_stand_ins(tmp_path)
    put_the_real_toolkit_command_on_path(bin_dir, "qp-ovms-config")
    return bin_dir


def run_app(
    tmp_path: Path, bin_dir: Path, component: str, **env: str
) -> subprocess.CompletedProcess[str]:
    full = stand_in_env(
        tmp_path,
        bin_dir,
        **model_server_env("ovms", component),
        QP_WORKFLOW_FILE=os.environ["QP_WORKFLOW_FILE"],
        PYTHONPATH=str(TOOLKIT_ROOT / "src"),
        **env,
    )
    return subprocess.run(
        ["bash", str(APP_SH)], env=full, capture_output=True, text=True, timeout=60
    )


def run_reranker(
    tmp_path: Path, bin_dir: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    models = tmp_path / "models"
    write_ovms_export(models, "reranker", "fixture-reranker")
    return run_app(
        tmp_path, bin_dir, "reranker", QP_OVMS_MODELS_DIR=str(models), **PINNED, **env
    )


def run_llm(
    tmp_path: Path, bin_dir: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    write_fake(
        bin_dir / "qp-served-models",
        tmp_path / "qp-served-models.log",
        f"echo '{SERVED_LLM}'\n",
    )
    return run_app(
        tmp_path,
        bin_dir,
        "llm",
        QP_OVMS_MODELS_DIR=str(tmp_path / "llm-models"),
        QP_COMPONENTS__LLM__SOURCE_ID=LLM_SOURCE_ID,
        **env,
    )


def ovms_argv(tmp_path: Path) -> list[str]:
    [started] = log(tmp_path, "ovms").splitlines()
    return started.split()


def test_a_default_rerankers_app_runs_todays_command_line(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_reranker(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    argv = ovms_argv(tmp_path)
    assert argv[:3] == ["--rest_port", "8888", "--config_path"]
    assert len(argv) == 4
    assert argv[3].endswith("/config_reranker.json")


def test_a_default_llms_app_runs_the_one_pass_command_line(
    tmp_path: Path, bin_dir: Path
) -> None:
    # The LLM's defaults read each prompt in one pass; no server flag follows.
    result = run_llm(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    assert ovms_argv(tmp_path) == [
        *("--rest_port", "8888", "--source_model", LLM_SOURCE_ID),
        *("--model_name", "fixture-chat"),
        *("--model_repository_path", str(tmp_path / "llm-models")),
        *("--task", "text_generation", "--target_device", "CPU"),
        *("--dynamic_split_fuse", "false", "--max_num_batched_tokens", "8192"),
    ]


def test_rest_workers_reaches_ovms_on_a_rerankers_app(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_reranker(tmp_path, bin_dir, QP_OVMS_REST_WORKERS="4")

    assert result.returncode == 0, result.stderr
    assert ovms_argv(tmp_path)[-2:] == ["--rest_workers", "4"]


def test_the_pass_through_reaches_ovms_on_an_llms_app(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_llm(
        tmp_path,
        bin_dir,
        QP_OVMS_ARGS="--log_level DEBUG",
        QP_OVMS_MAX_NUM_SEQS="64",
    )

    assert result.returncode == 0, result.stderr
    # The LLM's flags first, then the server flags.
    assert ovms_argv(tmp_path)[-4:] == ["--max_num_seqs", "64", "--log_level", "DEBUG"]


def bare_path(tmp_path: Path) -> Path:
    """System tools and a fake ``ovms``: no ``uv``, no ``qp-ovms-config``."""
    bare = tmp_path / "bare-bin"
    bare.mkdir()
    for name in ("bash", "env", "cat", "dirname", "mkdir", "chmod", "id", "mktemp"):
        found = shutil.which(name)
        assert found, f"{name} is needed on PATH to run app.sh"
        (bare / name).symlink_to(found)
    write_fake(bare / "ovms", tmp_path / "ovms.log")
    return bare


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("QP_OVMS_REST_WORKERS", "4"),
        ("QP_OVMS_ARGS", "--log_level DEBUG"),
        ("QP_OVMS_GRAPH_QUEUE_MAX_SIZE", "4"),
        ("QP_OVMS_EMBEDDINGS_TRUNCATE", "true"),
        ("QP_OVMS_OPENVINO__HYPER_THREADING", "false"),
        ("QP_OVMS_START_TIMEOUT_S", "30"),
    ],
)
def test_a_flag_setting_without_qp_ovms_config_stops_the_app(
    tmp_path: Path, variable: str, value: str
) -> None:
    models = tmp_path / "models"
    write_ovms_export(models, "reranker", "fixture-reranker")
    env = stand_in_env(
        tmp_path,
        bare_path(tmp_path),
        **model_server_env("ovms", "reranker"),
        QP_OVMS_MODELS_DIR=str(models),
        **{variable: value},
    )

    result = subprocess.run(
        ["bash", str(APP_SH)],
        env=env | {"PATH": str(tmp_path / "bare-bin")},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    [line] = [x for x in result.stderr.splitlines() if x.startswith("app.sh[")]
    assert f"{variable}={value}" in line
    assert "qp-ovms-config" in line
    assert "apps.reranker.settings" in line
    assert log(tmp_path, "ovms") == ""
