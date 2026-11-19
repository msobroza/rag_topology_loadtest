"""domino/app.sh as an OVMS model server of an LLM: OVMS pulls and serves the model.

No export and no calibration: the script asks ``qp-served-models`` the
component's type, source id and alias, and ``qp-ovms-config --serve-llm`` its
batching flags, then starts ``ovms`` with them. The stand-ins of
tests/helpers/launch_script_stand_ins.py record what would have run; here a
``qp-served-models`` stand-in prints an LLM's line, and a ``qp-ovms-config``
stand-in prints the flags of FAKE_LLM_FLAGS.
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
from repo_paths import TOOLKIT_ROOT

SERVED_MODELS = (
    "embedder embedder fixture-org/fixture-embedder fixture-embedder fp32",
    "llm llm OpenVINO/Fixture-Chat-int4-ov fixture-chat fp32",
)
# What the qp-ovms-config stand-in does for --serve-llm: print FAKE_LLM_FLAGS.
LLM_FLAGS_STAND_IN = 'echo "${FAKE_LLM_FLAGS:-}"\n'


def served_models_printing(*lines: str) -> str:
    """The body of a ``qp-served-models`` stand-in that prints ``lines``."""
    printed = "\n".join(lines)
    return f"cat <<'EOF'\n{printed}\nEOF\n"


def write_llm_stand_ins(bin_dir: Path, tmp_path: Path) -> None:
    """``qp-served-models`` and ``qp-ovms-config`` as an LLM's App asks them."""
    write_fake(
        bin_dir / "qp-served-models",
        tmp_path / "qp-served-models.log",
        served_models_printing(*SERVED_MODELS),
    )
    write_fake(
        bin_dir / "qp-ovms-config", tmp_path / "qp-ovms-config.log", LLM_FLAGS_STAND_IN
    )


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """The stand-ins, with an LLM's ``qp-served-models`` and ``qp-ovms-config``."""
    bin_dir = make_stand_ins(tmp_path)
    write_llm_stand_ins(bin_dir, tmp_path)
    return bin_dir


def run_app(
    tmp_path: Path, bin_dir: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(APP_SH)],
        env=stand_in_env(tmp_path, bin_dir, **model_server_env("ovms", "llm"), **env),
        capture_output=True,
        text=True,
        timeout=60,
    )


def llm_argv(models: Path) -> str:
    """The ``ovms`` arguments of the LLM after its REST port, at OVMS's defaults."""
    return (
        "--source_model OpenVINO/Fixture-Chat-int4-ov --model_name fixture-chat "
        f"--model_repository_path {models} --task text_generation "
        "--target_device CPU"
    )


def bare_path(tmp_path: Path, *tools: str) -> Path:
    """A PATH folder with only ``tools`` linked from this machine, no uv, no qp-*."""
    bare = tmp_path / "bare-bin"
    bare.mkdir()
    for name in ("bash", "env", "dirname", "cat", *tools):
        found = shutil.which(name)
        assert found, f"{name} is needed on PATH to run app.sh"
        (bare / name).symlink_to(found)
    write_fake(bare / "ovms", tmp_path / "ovms.log")
    return bare


def run_bare(
    tmp_path: Path, bare: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    full = stand_in_env(tmp_path, bare, **model_server_env("ovms", "llm"), **env)
    return subprocess.run(
        ["bash", str(APP_SH)],
        env=full | {"PATH": str(bare)},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_an_llm_is_pulled_and_served_by_ovms_with_no_export_nor_calibration(
    tmp_path: Path, bin_dir: Path
) -> None:
    models = tmp_path / "llm-models"

    result = run_app(tmp_path, bin_dir, QP_OVMS_MODELS_DIR=str(models))

    assert result.returncode == 0, result.stderr
    # The qp-ovms-config stand-in prints no flag, so none is passed.
    assert log(tmp_path, "ovms").strip() == f"--rest_port 8888 {llm_argv(models)}"
    assert log(tmp_path, "qp-ovms-config").strip() == "--serve-llm llm"
    assert log(tmp_path, "uv") == ""
    assert not (models / "config_llm.json").exists()


def test_the_llms_batching_flags_reach_ovms(tmp_path: Path, bin_dir: Path) -> None:
    models = tmp_path / "llm-models"

    result = run_app(
        tmp_path,
        bin_dir,
        QP_OVMS_MODELS_DIR=str(models),
        FAKE_LLM_FLAGS="--dynamic_split_fuse false --max_num_batched_tokens 8192",
    )

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "ovms").strip() == (
        f"--rest_port 8888 {llm_argv(models)} --dynamic_split_fuse false "
        "--max_num_batched_tokens 8192"
    )


@pytest.mark.parametrize(
    ("env", "flags"),
    [
        ({}, " --dynamic_split_fuse false --max_num_batched_tokens 8192"),
        (
            {"QP_OVMS_DYNAMIC_SPLIT_FUSE": "true", "QP_OVMS_MAX_BATCHED_TOKENS": "0"},
            "",
        ),
    ],
    ids=["defaults", "ovms-own"],
)
def test_a_default_llms_app_reads_each_prompt_in_one_pass(
    tmp_path: Path, bin_dir: Path, env: dict[str, str], flags: str
) -> None:
    # The real qp-ovms-config, so the flags are the settings' defaults.
    put_the_real_toolkit_command_on_path(bin_dir, "qp-ovms-config")
    models = tmp_path / "llm-models"

    result = run_app(
        tmp_path,
        bin_dir,
        QP_OVMS_MODELS_DIR=str(models),
        QP_COMPONENTS__LLM__SOURCE_ID="OpenVINO/Fixture-Chat-int4-ov",
        QP_WORKFLOW_FILE=os.environ["QP_WORKFLOW_FILE"],
        **env,
    )

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "ovms").strip() == (
        f"--rest_port 8888 {llm_argv(models)}{flags}"
    )


def test_the_llm_is_served_behind_the_prefix_proxy(
    tmp_path: Path, bin_dir: Path
) -> None:
    models = tmp_path / "llm-models"

    result = run_app(
        tmp_path, bin_dir, QP_OVMS_MODELS_DIR=str(models), QP_OVMS_PREFIX_PROXY="1"
    )

    assert result.returncode == 1  # the stand-ins exit at once
    assert log(tmp_path, "ovms").strip() == (
        f"--rest_bind_address 127.0.0.1 --rest_port 9000 {llm_argv(models)}"
    )
    assert log(tmp_path, "nginx").strip() == "-c /tmp/ovms-proxy.conf -g daemon off;"


def test_an_llms_app_installs_the_project_to_learn_its_model(tmp_path: Path) -> None:
    bare = bare_path(tmp_path, "mkdir", "chmod", "cp")
    templates = tmp_path / "templates"
    write_llm_stand_ins(templates, tmp_path)
    write_fake(
        bare / "uv",
        tmp_path / "uv.log",
        'if [ "$1" = venv ]; then mkdir -p "$2/bin"\n'
        f'  cp "{templates}/qp-served-models" "{templates}/qp-ovms-config" "$2/bin/"\n'
        "fi\n",
    )
    models = tmp_path / "llm-models"

    result = run_bare(tmp_path, bare, QP_OVMS_MODELS_DIR=str(models))

    assert result.returncode == 0, result.stderr
    venv = tmp_path / "home" / "qp-venv"
    assert log(tmp_path, "uv").splitlines() == [
        f"venv {venv} --python 3.11",
        f"pip install --python {venv}/bin/python {TOOLKIT_ROOT}",
    ]
    assert log(tmp_path, "ovms").strip() == f"--rest_port 8888 {llm_argv(models)}"


def test_an_llms_app_without_the_project_or_uv_names_both(tmp_path: Path) -> None:
    result = run_bare(tmp_path, bare_path(tmp_path))

    assert result.returncode == 1
    message = result.stderr.strip().splitlines()[-1]
    assert "qp-served-models" in message
    assert "'uv'" in message
    assert log(tmp_path, "ovms") == ""


def test_an_llms_app_creates_its_models_folder_before_ovms_starts(
    tmp_path: Path, bin_dir: Path
) -> None:
    # Nobody has seen OVMS create the folder itself, so the script makes it first.
    models = tmp_path / "not-yet" / "llm-models"
    seen = tmp_path / "folder-at-ovms-start"
    write_fake(
        bin_dir / "ovms",
        tmp_path / "ovms.log",
        f'[ -d "{models}" ] && echo yes > "{seen}"\n',
    )

    result = run_app(tmp_path, bin_dir, QP_OVMS_MODELS_DIR=str(models))

    assert result.returncode == 0, result.stderr
    assert seen.read_text().strip() == "yes"


def test_a_models_folder_that_cannot_be_created_stops_the_llms_app(
    tmp_path: Path, bin_dir: Path
) -> None:
    regular_file = tmp_path / "a-file"
    regular_file.write_text("")
    models = regular_file / "llm-models"

    result = run_app(tmp_path, bin_dir, QP_OVMS_MODELS_DIR=str(models))

    assert result.returncode == 1
    message = result.stderr.strip().splitlines()[-1]
    assert message.startswith("app.sh[model-server ovms llm]: ")
    assert str(models) in message
    assert "QP_OVMS_MODELS_DIR" in message
    assert log(tmp_path, "ovms") == ""


def test_an_llms_app_without_ovms_names_the_environment_that_runs_it(
    tmp_path: Path,
) -> None:
    # The prebuilt ovms image cannot start an LLM; the generic environment can.
    bare = bare_path(tmp_path)
    (bare / "ovms").unlink()
    write_llm_stand_ins(bare, tmp_path)

    result = run_bare(tmp_path, bare)

    assert result.returncode == 1
    message = result.stderr.strip().splitlines()[-1]
    assert "QP_BOOTSTRAP=1" in message
    assert "QP_OVMS_BUNDLE" in message
    assert "--target ovms" not in message


@pytest.mark.parametrize("failing", ["qp-served-models", "qp-ovms-config"])
def test_a_failing_toolkit_command_stops_the_llms_app(
    tmp_path: Path, bin_dir: Path, failing: str
) -> None:
    write_fake(
        bin_dir / failing,
        tmp_path / f"{failing}.log",
        f'echo "{failing}: cannot serve the llm: boom" >&2\nexit 1\n',
    )

    result = run_app(tmp_path, bin_dir, QP_OVMS_MODELS_DIR=str(tmp_path / "m"))

    assert result.returncode == 1
    assert f"{failing}: cannot serve the llm: boom" in result.stderr
    assert log(tmp_path, "ovms") == ""


def test_an_llms_config_left_in_its_folder_stops_the_app_naming_the_file(
    tmp_path: Path, bin_dir: Path
) -> None:
    # With a config there, the script takes the LLM for an export to serve;
    # the real qp-ovms-config knows it is an LLM and says which file is in the way.
    put_the_real_toolkit_command_on_path(bin_dir, "qp-ovms-config")
    models = tmp_path / "llm-models"
    models.mkdir()
    (models / "config_llm.json").write_text("{}")
    (models / "weight-format").write_text("fp32\n")

    result = run_app(
        tmp_path,
        bin_dir,
        QP_OVMS_MODELS_DIR=str(models),
        QP_COMPONENTS__LLM__SOURCE_ID="OpenVINO/Fixture-Chat-int4-ov",
        QP_WORKFLOW_FILE=os.environ["QP_WORKFLOW_FILE"],
    )

    assert result.returncode == 1
    assert f"{models / 'config_llm.json'}" in result.stderr
    assert "an LLM is served without one" in result.stderr
    assert log(tmp_path, "ovms") == ""
