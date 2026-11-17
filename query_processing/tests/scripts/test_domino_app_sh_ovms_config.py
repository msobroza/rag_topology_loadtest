"""domino/app.sh: where the config an OVMS model server starts ``ovms`` with comes from.

``qp-ovms-config`` prints it when it is on PATH (test_domino_app_sh.py covers
that); here it is not, so app.sh installs the project with ``uv``, or, without
``uv``, serves the export's own config unless a ``QP_OVMS_*`` pin asks for more.
PATH holds only fakes and links to the few system tools the scripts use, so no
``uv``, ``ovms`` or ``qp-ovms-config`` installed on this machine is reached.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from launch_script_stand_ins import (
    APP_SH,
    log,
    model_server_env,
    models_dir,
    served_config,
    write_fake,
    write_ovms_config_stand_in,
)
from repo_paths import TOOLKIT_ROOT

SYSTEM_TOOLS = ("bash", "env", "cat", "dirname", "mkdir", "chmod", "id", "mktemp")


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """System tools and a fake ``ovms``; no ``uv``, no ``qp-ovms-config``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in SYSTEM_TOOLS:
        found = shutil.which(name)
        assert found, f"{name} is needed on PATH to run app.sh"
        (bin_dir / name).symlink_to(found)
    write_fake(bin_dir / "ovms", tmp_path / "ovms.log")
    return bin_dir


def with_uv(tmp_path: Path, bin_dir: Path) -> None:
    """A fake ``uv`` that chatters on stdout; ``uv venv <dir>`` puts a
    ``qp-ovms-config`` stand-in (logging as ``venv-qp-ovms-config``) in it."""
    template = tmp_path / "templates" / "qp-ovms-config"
    write_ovms_config_stand_in(template, tmp_path, name="venv-qp-ovms-config")
    write_fake(
        bin_dir / "uv",
        tmp_path / "uv.log",
        'echo "Resolved 42 packages"\n'
        'if [ "$1" = venv ]; then mkdir -p "$2/bin"\n'
        f'  cat "{template}" > "$2/bin/qp-ovms-config"\n'
        '  chmod +x "$2/bin/qp-ovms-config"\nfi\n',
    )


def run_app(
    tmp_path: Path, bin_dir: Path, *, recorded: str = "fp32", **env: str
) -> subprocess.CompletedProcess[str]:
    home, temporary = tmp_path / "home", tmp_path / "tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    models = models_dir(tmp_path, recorded)
    full_env = {
        "PATH": str(bin_dir),
        "HOME": str(home),
        "TMPDIR": str(temporary),
        **model_server_env("ovms", "reranker"),
        "QP_OVMS_MODELS_DIR": str(models),
        **env,
    }
    return subprocess.run(
        ["bash", str(APP_SH)],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_with_uv_the_project_is_installed_and_its_config_served(
    tmp_path: Path, bin_dir: Path
) -> None:
    with_uv(tmp_path, bin_dir)

    result = run_app(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    venv = tmp_path / "home" / "qp-venv"
    assert log(tmp_path, "uv").splitlines() == [
        f"venv {venv} --python 3.11",
        f"pip install --python {venv}/bin/python {TOOLKIT_ROOT}",
    ]
    config = served_config(tmp_path, "venv-qp-ovms-config")
    assert config.startswith(str(tmp_path / "tmp"))  # a fresh temporary folder
    # uv's chatter went to stderr, so the captured path is the path alone.
    assert log(tmp_path, "ovms").strip() == f"--rest_port 8888 --config_path {config}"
    assert "Resolved 42 packages" in result.stderr


def test_without_uv_the_export_is_served_as_exported_and_says_so(
    tmp_path: Path, bin_dir: Path
) -> None:
    result = run_app(tmp_path, bin_dir)

    assert result.returncode == 0, result.stderr
    models = tmp_path / "models"
    assert log(tmp_path, "ovms").strip() == (
        f"--rest_port 8888 --config_path {models}/config_reranker.json"
    )
    [line] = [x for x in result.stderr.splitlines() if "qp-ovms-config" in x]
    assert line.startswith("app.sh[model-server ovms reranker]: ")
    assert "not calibrated" in line


@pytest.mark.parametrize("pin", ["QP_OVMS_NUM_STREAMS", "QP_OVMS_THREADS"])
def test_without_uv_a_pin_that_cannot_be_applied_stops_the_app(
    tmp_path: Path, bin_dir: Path, pin: str
) -> None:
    result = run_app(tmp_path, bin_dir, **{pin: "2"})

    assert result.returncode == 1
    assert "QP_OVMS_NUM_STREAMS / QP_OVMS_THREADS are set" in result.stderr
    assert log(tmp_path, "ovms") == ""


@pytest.mark.parametrize(
    ("option", "value"),
    [("QP_OVMS_DYNAMIC_SPLIT_FUSE", "false"), ("QP_OVMS_MAX_BATCHED_TOKENS", "512")],
)
def test_without_uv_an_llms_option_on_a_reranker_stops_the_app(
    tmp_path: Path, bin_dir: Path, option: str, value: str
) -> None:
    result = run_app(tmp_path, bin_dir, **{option: value})

    assert result.returncode == 1
    [line] = [x for x in result.stderr.splitlines() if x.startswith("app.sh[")]
    assert option in line
    assert "apply to an LLM" in line
    assert log(tmp_path, "ovms") == ""


def test_a_failing_qp_ovms_config_stops_the_app_and_names_the_pins(
    tmp_path: Path, bin_dir: Path
) -> None:
    write_fake(
        bin_dir / "qp-ovms-config",
        tmp_path / "qp-ovms-config.log",
        'echo "qp-ovms-config: cannot write the reranker config: boom" >&2\nexit 1\n',
    )

    result = run_app(tmp_path, bin_dir)

    assert result.returncode == 1
    assert "cannot write the reranker config: boom" in result.stderr
    [line] = [x for x in result.stderr.splitlines() if x.startswith("app.sh[")]
    assert "QP_OVMS_NUM_STREAMS" in line
    assert "QP_OVMS_THREADS" in line
    assert log(tmp_path, "ovms") == ""


@pytest.mark.parametrize(
    ("recorded", "env", "asked"),
    [
        ("int8", {}, "fp32"),  # unset: the default, all the script can see
        ("fp32", {"QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT": "int8"}, "int8"),
    ],
    ids=["int8-folder-default", "fp32-folder-int8-asked"],
)
def test_without_qp_ovms_config_the_script_checks_the_weight_format_itself(
    tmp_path: Path, bin_dir: Path, recorded: str, env: dict[str, str], asked: str
) -> None:
    # No command reads the settings here, so the variable, else the default,
    # is what is asked.
    result = run_app(tmp_path, bin_dir, recorded=recorded, **env)

    assert result.returncode == 1
    assert log(tmp_path, "ovms") == ""
    [line] = [x for x in result.stderr.splitlines() if "weight format" in x]
    assert line.startswith("app.sh[model-server ovms reranker]: ")
    assert str(tmp_path / "models") in line
    assert f"weight format {recorded}, but {asked} was asked" in line
    assert f"set QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT={recorded}" in line
    assert "point QP_OVMS_MODELS_DIR at another folder" in line
