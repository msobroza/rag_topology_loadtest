"""domino/app.sh with RAG_ROLE=ovms-generator: OVMS pulls and serves the generator.

No export and no calibration: the Role evaluates ``rag-served-models`` for the
model and its two OVMS options, then starts ``ovms`` with them. The stand-ins of
tests/helpers/launch_script_stand_ins.py record what would have run; a
``rag-served-models`` stand-in prints the four ``GENERATOR_*`` assignments.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from launch_script_stand_ins import (
    APP_SH,
    log,
    make_stand_ins,
    stand_in_env,
    write_fake,
)


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """Fake binaries that append their argv (and nginx its config) to <name>.log."""
    return make_stand_ins(tmp_path)


def run_app(
    tmp_path: Path, bin_dir: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(APP_SH)],
        env=stand_in_env(tmp_path, bin_dir, **env),
        capture_output=True,
        text=True,
        timeout=60,
    )


GENERATOR_DEFAULTS = (
    "GENERATOR_SOURCE_MODEL=OpenVINO/Qwen3.5-2B-int4-ov",
    "GENERATOR_MODEL_NAME=gpt-4o-mini",
    "GENERATOR_DYNAMIC_SPLIT_FUSE=true",
    "GENERATOR_MAX_BATCHED_TOKENS=''",
)


def served_models_printing(*lines: str) -> str:
    """The body of a ``rag-served-models`` stand-in that prints ``lines``."""
    return "".join(f"echo {shlex.quote(line)}\n" for line in lines)


def generator_argv(models: Path) -> str:
    """The ``ovms`` arguments of the generator after its REST port, by default.

    No ``--dynamic_split_fuse``: at ``true``, OVMS's own default, the flag is left
    out, so the default command line is the one run on a machine.
    """
    return (
        "--source_model OpenVINO/Qwen3.5-2B-int4-ov --model_name gpt-4o-mini "
        f"--model_repository_path {models} --task text_generation "
        "--target_device CPU"
    )


def bare_path(tmp_path: Path, *tools: str) -> Path:
    """A PATH folder with only ``tools`` linked from this machine, no uv, no rag-*."""
    bare = tmp_path / "bare-bin"
    bare.mkdir()
    for name in ("bash", "env", "dirname", "cat", *tools):
        found = shutil.which(name)
        assert found, f"{name} is needed on PATH to run app.sh"
        (bare / name).symlink_to(found)
    write_fake(bare / "ovms", tmp_path / "ovms.log")
    return bare


def test_the_generator_role_has_ovms_pull_and_serve_the_model(
    tmp_path: Path, bin_dir: Path
) -> None:
    write_fake(
        bin_dir / "rag-served-models",
        tmp_path / "rag-served-models.log",
        served_models_printing(*GENERATOR_DEFAULTS),
    )
    models = tmp_path / "generator-models"

    result = run_app(
        tmp_path, bin_dir, RAG_ROLE="ovms-generator", RAG_OVMS_MODELS_DIR=str(models)
    )

    assert result.returncode == 0, result.stderr
    # Split fuse on and 0 batched tokens are OVMS's own defaults: both flags are
    # left out altogether.
    assert log(tmp_path, "ovms").strip() == (
        f"--rest_port 8888 {generator_argv(models)}"
    )
    assert (log(tmp_path, "rag-ovms-config"), log(tmp_path, "uv")) == ("", "")


def test_the_generators_tuning_reaches_ovms(tmp_path: Path, bin_dir: Path) -> None:
    lines = (
        *GENERATOR_DEFAULTS[:2],
        "GENERATOR_DYNAMIC_SPLIT_FUSE=false",
        "GENERATOR_MAX_BATCHED_TOKENS=8192",
    )
    write_fake(
        bin_dir / "rag-served-models",
        tmp_path / "rag-served-models.log",
        served_models_printing(*lines),
    )
    models = tmp_path / "generator-models"

    result = run_app(
        tmp_path, bin_dir, RAG_ROLE="ovms-generator", RAG_OVMS_MODELS_DIR=str(models)
    )

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "ovms").strip() == (
        f"--rest_port 8888 {generator_argv(models)} --dynamic_split_fuse false "
        "--max_num_batched_tokens 8192"
    )


def test_the_generator_is_served_behind_the_prefix_proxy(
    tmp_path: Path, bin_dir: Path
) -> None:
    write_fake(
        bin_dir / "rag-served-models",
        tmp_path / "rag-served-models.log",
        served_models_printing(*GENERATOR_DEFAULTS),
    )
    models = tmp_path / "generator-models"

    result = run_app(
        tmp_path,
        bin_dir,
        RAG_ROLE="ovms-generator",
        RAG_OVMS_MODELS_DIR=str(models),
        RAG_OVMS_PREFIX_PROXY="1",
    )

    assert result.returncode == 1  # the stand-ins exit at once
    assert log(tmp_path, "ovms").strip() == (
        f"--rest_bind_address 127.0.0.1 --rest_port 9000 {generator_argv(models)}"
    )
    assert log(tmp_path, "nginx").strip() == "-c /tmp/ovms-proxy.conf -g daemon off;"


def test_the_generator_role_installs_the_project_to_learn_its_model(
    tmp_path: Path,
) -> None:
    bare = bare_path(tmp_path, "mkdir", "chmod")
    venv_command = served_models_printing(*GENERATOR_DEFAULTS).replace("'", "'\\''")
    write_fake(
        bare / "uv",
        tmp_path / "uv.log",
        'if [ "$1" = venv ]; then mkdir -p "$2/bin"\n'
        f"  printf '#!/usr/bin/env bash\\n%s' '{venv_command}'"
        ' > "$2/bin/rag-served-models"; chmod +x "$2/bin/rag-served-models"\n'
        "fi\n",
    )
    models = tmp_path / "generator-models"
    env = stand_in_env(
        tmp_path, bare, RAG_ROLE="ovms-generator", RAG_OVMS_MODELS_DIR=str(models)
    )

    result = subprocess.run(
        ["bash", str(APP_SH)],
        env=env | {"PATH": str(bare)},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    venv = tmp_path / "home" / "rag-venv"
    assert log(tmp_path, "uv").splitlines() == [
        f"venv {venv} --python 3.11",
        f"pip install --python {venv}/bin/python .",
    ]
    assert log(tmp_path, "ovms").strip() == (
        f"--rest_port 8888 {generator_argv(models)}"
    )


def test_the_generator_role_without_the_project_or_uv_names_both(
    tmp_path: Path,
) -> None:
    bare = bare_path(tmp_path)
    env = stand_in_env(tmp_path, bare, RAG_ROLE="ovms-generator")

    result = subprocess.run(
        ["bash", str(APP_SH)],
        env=env | {"PATH": str(bare)},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "rag-served-models" in result.stderr
    assert "uv" in result.stderr
    assert log(tmp_path, "ovms") == ""


def test_the_generator_role_creates_its_models_folder_before_ovms_starts(
    tmp_path: Path, bin_dir: Path
) -> None:
    # Nobody has seen OVMS create the folder itself, so app.sh makes it first.
    models = tmp_path / "not-yet" / "generator-models"
    write_fake(
        bin_dir / "rag-served-models",
        tmp_path / "rag-served-models.log",
        served_models_printing(*GENERATOR_DEFAULTS),
    )
    seen = tmp_path / "folder-at-ovms-start"
    write_fake(
        bin_dir / "ovms",
        tmp_path / "ovms.log",
        f'[ -d "{models}" ] && echo yes > "{seen}"\n',
    )

    result = run_app(
        tmp_path, bin_dir, RAG_ROLE="ovms-generator", RAG_OVMS_MODELS_DIR=str(models)
    )

    assert result.returncode == 0, result.stderr
    assert seen.read_text().strip() == "yes"


def test_a_models_folder_that_cannot_be_created_stops_the_generator_role(
    tmp_path: Path, bin_dir: Path
) -> None:
    write_fake(
        bin_dir / "rag-served-models",
        tmp_path / "rag-served-models.log",
        served_models_printing(*GENERATOR_DEFAULTS),
    )
    regular_file = tmp_path / "a-file"
    regular_file.write_text("")
    models = regular_file / "generator-models"

    result = run_app(
        tmp_path, bin_dir, RAG_ROLE="ovms-generator", RAG_OVMS_MODELS_DIR=str(models)
    )

    assert result.returncode == 1
    message = result.stderr.strip().splitlines()[-1]
    assert message.startswith("app.sh[ovms-generator]: ")
    assert str(models) in message
    assert "RAG_OVMS_MODELS_DIR" in message
    assert log(tmp_path, "ovms") == ""


def test_the_generator_role_without_ovms_names_the_environment_that_runs_it(
    tmp_path: Path,
) -> None:
    # The prebuilt ovms image cannot start this Role; the generic environment can.
    bare = bare_path(tmp_path)
    (bare / "ovms").unlink()
    env = stand_in_env(tmp_path, bare, RAG_ROLE="ovms-generator")

    result = subprocess.run(
        ["bash", str(APP_SH)],
        env=env | {"PATH": str(bare)},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    message = result.stderr.strip().splitlines()[-1]
    assert "RAG_BOOTSTRAP=1" in message
    assert "RAG_OVMS_BUNDLE" in message
    assert "--target ovms" not in message
