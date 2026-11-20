"""An App that installs takes doclang's core from a directory of wheels.

doclang's core (``doclang-domain-core``) is private and on no index, and an App
has no clone of it: ``domino/app.sh``'s one install function finds it among the
wheels of the directory ``QP_BOOTSTRAP_WHEELS`` names, which the deploy setting
``bootstrap_wheels`` gives every App, and everything else on the index. Without
a wheel there, the App stops before it installs, in one line. The stand-ins of
tests/helpers/launch_script_stand_ins.py record what ``uv`` was asked; nothing
is installed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from launch_script_stand_ins import (
    APP_SH,
    WHEELS_VARIABLE,
    log,
    make_stand_ins,
    model_server_env,
    stand_in_env,
    wheels_env,
)
from repo_paths import CORE_ROOT, TOOLKIT_ROOT

# The two roles that install the toolkit when asked to (QP_BOOTSTRAP=1): the
# workflow app, and a model server of the fastapi kind, whose bootstrap is the
# workflow app's install.
INSTALLING = [{"QP_ROLE": "workflow"}, model_server_env("fastapi", "reranker")]


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
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


@pytest.mark.parametrize("env", INSTALLING, ids=["workflow", "fastapi"])
def test_the_install_finds_doclangs_core_among_the_wheels_of_the_directory(
    tmp_path: Path, bin_dir: Path, env: dict[str, str]
) -> None:
    wheels = wheels_env(tmp_path)

    result = run_app(tmp_path, bin_dir, **env, QP_BOOTSTRAP="1", **wheels)

    assert result.returncode == 0, result.stderr
    venv = tmp_path / "home" / "qp-venv"
    assert log(tmp_path, "uv").splitlines()[1] == (
        f"pip install --python {venv}/bin/python "
        f"--find-links {wheels[WHEELS_VARIABLE]} {CORE_ROOT} {TOOLKIT_ROOT}"
    )


def a_directory_with_another_wheel(tmp_path: Path) -> str:
    """A directory holding a wheel, but none of doclang's core."""
    wheels = tmp_path / "other wheels"
    wheels.mkdir()
    (wheels / "doclang_domain_ingestion-0.1.0-py3-none-any.whl").touch()
    return str(wheels)


@pytest.mark.parametrize(
    "where",
    ["unset", "empty", "missing", "another wheel"],
)
@pytest.mark.parametrize("env", INSTALLING, ids=["workflow", "fastapi"])
def test_without_a_wheel_of_doclangs_core_the_app_stops_before_it_installs(
    tmp_path: Path, bin_dir: Path, env: dict[str, str], where: str
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    directories = {
        "unset": "",
        "empty": str(empty),
        "missing": str(tmp_path / "no such directory"),
        "another wheel": a_directory_with_another_wheel(tmp_path),
    }
    given = {WHEELS_VARIABLE: directories[where]} if where != "unset" else {}

    result = run_app(tmp_path, bin_dir, **env, QP_BOOTSTRAP="1", **given)

    assert result.returncode == 1
    [line] = result.stderr.splitlines()
    assert f"{WHEELS_VARIABLE}='{directories[where]}'" in line
    assert "doclang_domain_core-*.whl" in line
    assert "bootstrap_wheels" in line  # the deploy setting that names it
    assert log(tmp_path, "uv") == ""  # nothing installed, not even the venv
    assert log(tmp_path, "venv-qp-serve") + log(tmp_path, "qp-serve") == ""


@pytest.mark.parametrize("env", INSTALLING, ids=["workflow", "fastapi"])
def test_an_app_not_asked_to_install_needs_no_wheel(
    tmp_path: Path, bin_dir: Path, env: dict[str, str]
) -> None:
    # The prebuilt image already holds doclang's core: no directory is needed.
    result = run_app(tmp_path, bin_dir, **env)

    assert result.returncode == 0, result.stderr
    assert log(tmp_path, "uv") == ""
