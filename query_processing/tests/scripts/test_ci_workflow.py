"""The CI workflow clones document_doclang beside the checkout, then installs as the laptop does.

``.github/workflows/checks.yml`` cannot be run here: its clone step is read from
the file and run under ``sh -e``, the shell GitHub gives a run step of a
container job, with ``git`` and ``ssh`` stand-ins that record what they were
asked; nothing reaches the network. Its install line is held to what the root
Makefile's ``install-dev`` installs. What only a run on GitHub shows: that the
secret clones, and that the image's ``ssh`` accepts the pinned host key.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from repo_paths import REPO_ROOT

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "checks.yml"
CLONE_STEP = "Clone document_doclang beside the checkout"
CLONE = "../document_doclang"
# GitHub's published fingerprint of its ed25519 host key (docs.github.com,
# "GitHub's SSH key fingerprints"), against which the pinned key is held.
GITHUB_ED25519 = "SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU"


def check_steps() -> list[dict[str, Any]]:
    """The steps of the workflow's ``check`` job, as the file holds them."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["check"]["steps"]


def the_clone_step() -> dict[str, Any]:
    [step] = [step for step in check_steps() if step.get("name") == CLONE_STEP]
    return step


def run_the_clone_step(
    where: Path, key: str, *, git_answers: int = 0
) -> subprocess.CompletedProcess[str]:
    """The clone step's script under ``sh -e``, ``git`` and ``ssh`` stood in.

    ``git`` logs its argv and ``GIT_SSH_COMMAND`` to ``where/git.log`` and exits
    with ``git_answers``; ``ssh`` and ``chown`` log to theirs.
    """
    bin_dir = where / "bin"
    bin_dir.mkdir()
    for name, extra in [("git", 'echo "$GIT_SSH_COMMAND"'), ("ssh", ""), ("chown", "")]:
        (bin_dir / name).write_text(
            f'#!/bin/sh\n{{ echo "$@"; {extra or ":"}; }} >> "{where}/{name}.log"\n'
            f"exit {git_answers if name == 'git' else 0}\n",
            encoding="utf-8",
        )
        (bin_dir / name).chmod(0o755)
    step = the_clone_step()
    env = {**step["env"], "DOCLANG_DEPLOY_KEY": key}
    return subprocess.run(
        ["sh", "-e", "-c", step["run"]],
        cwd=where,
        env={**env, "PATH": f"{bin_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def logged(where: Path, name: str) -> str:
    path = where / f"{name}.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_without_the_secret_the_step_stops_in_one_line_before_any_ssh(
    tmp_path: Path,
) -> None:
    # A pull request from a fork gets no secret: the key was an empty file and
    # the clone failed on GitHub's answer, after the ssh.
    result = run_the_clone_step(tmp_path, key="")

    assert result.returncode != 0
    [line] = (result.stdout + result.stderr).splitlines()
    assert "DOCLANG_DEPLOY_KEY" in line
    assert "fork" in line
    assert logged(tmp_path, "git") + logged(tmp_path, "ssh") == ""


@pytest.mark.parametrize("git_answers", [0, 128], ids=["cloned", "clone-failed"])
def test_the_keys_file_is_removed_whether_the_clone_succeeds_or_not(
    tmp_path: Path, git_answers: int
) -> None:
    result = run_the_clone_step(tmp_path, key="a key", git_answers=git_answers)

    assert result.returncode == git_answers
    [clone, ssh_command] = logged(tmp_path, "git").splitlines()
    assert clone.endswith(f"document_doclang.git {CLONE}")
    # The key's file and the pinned host key's, both made by the step.
    files = re.findall(r"(?:-i |UserKnownHostsFile=)(\S+)", ssh_command)
    assert len(files) == 2, ssh_command
    assert [path for path in files if Path(path).exists()] == []
    assert "-o StrictHostKeyChecking=yes" in ssh_command
    assert "accept-new" not in ssh_command


def test_githubs_host_key_is_pinned_in_the_workflow() -> None:
    # Accepted on first use, a host answering for github.com would get the key.
    [host, kind, blob] = the_clone_step()["env"]["GITHUB_HOST_KEY"].split()
    digest = hashlib.sha256(base64.b64decode(blob)).digest()

    assert (host, kind) == ("github.com", "ssh-ed25519")
    fingerprint = "SHA256:" + base64.b64encode(digest).decode().rstrip("=")
    assert fingerprint == GITHUB_ED25519


def a_clone(where: Path) -> Path:
    """A stand-in clone of document_doclang: its core's manifest alone."""
    (where / "packages" / "core").mkdir(parents=True)
    (where / "packages" / "core" / "pyproject.toml").write_text("", encoding="utf-8")
    return where


def test_ci_installs_what_make_install_dev_installs_from_the_clone_beside(
    tmp_path: Path,
) -> None:
    # The two would drift: CI installs with uv, the Makefile with pip.
    clone = a_clone(tmp_path / "document_doclang")
    inherited = ("MAKEFLAGS", "MFLAGS", "DOCLANG_CLONE")
    env = {k: v for k, v in os.environ.items() if k not in inherited}
    make = subprocess.run(
        ["make", "-n", "install-dev", "BIN=", f"DOCLANG_CLONE={clone}"],
        cwd=REPO_ROOT,
        env={**env, "MAKELEVEL": "0"},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    [makefile_line] = make.stdout.splitlines()
    runs = [step.get("run", "") for step in check_steps()]

    [ci_line] = [run for run in runs if "uv pip install -e" in run]
    makefile_args = makefile_line.removeprefix("python -m pip install ")
    assert ci_line == "$AS_UBUNTU uv pip install " + makefile_args.replace(
        str(clone), CLONE
    )
