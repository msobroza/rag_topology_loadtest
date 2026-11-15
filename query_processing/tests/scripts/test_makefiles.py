"""The two Makefiles: the repository's hands the gates to the toolkit's.

``make -n`` prints the commands without running them; a recursive ``$(MAKE)``
still runs, under ``-n`` too, so the toolkit's commands are printed as well. The
tools' place is given as ``BIN``, as the root Makefile computes it from its
``.venv``, so the result does not depend on this machine.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from repo_paths import REPO_ROOT, TOOLKIT_ROOT

TOOLS = "/tools/bin/"  # never run: printed only
GATES = ("lint", "typecheck", "complexity", "deadcode", "deps", "test")


def dry_run(directory: Path, *args: str) -> list[str]:
    """The commands ``make -n <args>`` prints in ``directory``, one per line."""
    # A make running this suite would hand its own flags and variables down.
    env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS")}
    run = subprocess.run(
        ["make", "-n", *args],
        cwd=directory,
        env={**env, "MAKELEVEL": "0"},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    # GNU make 4 says where a recursive make enters and leaves; 3.81 does not.
    return [
        line.strip()
        for line in run.stdout.splitlines()
        if not re.match(r"\S*make(\[\d+\])?: (Entering|Leaving) directory", line)
    ]


@pytest.mark.parametrize("target", ["install", "install-dev", "check", *GATES])
def test_the_root_hands_each_gate_to_the_toolkit_with_the_tools_place(
    target: str,
) -> None:
    lines = dry_run(REPO_ROOT, target, f"BIN={TOOLS}")

    assert lines[0].endswith(f' -C {TOOLKIT_ROOT.name} {target} BIN="{TOOLS}"')


def test_install_dev_at_the_root_installs_the_toolkit_from_its_directory() -> None:
    lines = dry_run(REPO_ROOT, "install-dev", f"BIN={TOOLS}")

    assert lines[1:] == [f'{TOOLS}python -m pip install -e ".[dev]"']


def test_check_at_the_root_runs_the_toolkits_gates_in_order() -> None:
    from_root = dry_run(REPO_ROOT, "check", f"BIN={TOOLS}")[1:]
    from_toolkit = dry_run(TOOLKIT_ROOT, "check", f"BIN={TOOLS}")

    assert from_root == from_toolkit
    assert from_toolkit[0] == f"{TOOLS}ruff check src tests locustfile.py"
    assert from_toolkit[-1] == f"{TOOLS}pytest"
