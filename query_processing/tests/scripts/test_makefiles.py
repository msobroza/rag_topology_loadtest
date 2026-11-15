"""The repository's Makefile and the toolkit's: the root installs the toolkit and
the use case together, hands the gates to the toolkit's, then runs the use
case's tests.

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
# The RAG use case's directory, beside the toolkit's: the root installs both and
# runs the use case's tests. Named only: no test of the toolkit reads its files.
USE_CASE = "query_processing_rag_usecase"


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


@pytest.mark.parametrize("target", ["check", "format", *GATES])
def test_the_root_hands_each_gate_to_the_toolkit_with_the_tools_place(
    target: str,
) -> None:
    lines = dry_run(REPO_ROOT, target, f"BIN={TOOLS}")

    assert lines[0].endswith(f' -C {TOOLKIT_ROOT.name} {target} BIN="{TOOLS}"')


@pytest.mark.parametrize(
    ("target", "toolkit"),
    [("install", "./query_processing"), ("install-dev", '"./query_processing[dev]"')],
)
def test_the_root_installs_the_toolkit_and_the_use_case_in_one_command(
    target: str, toolkit: str
) -> None:
    # No package index has the toolkit: the use case's dependency on it is met
    # only when both directories are named, the toolkit first.
    lines = dry_run(REPO_ROOT, target, f"BIN={TOOLS}")

    assert lines == [f"{TOOLS}python -m pip install -e {toolkit} -e ./{USE_CASE}"]
    assert (REPO_ROOT / toolkit.strip('"').removesuffix("[dev]")).resolve() == (
        TOOLKIT_ROOT
    )


def test_the_toolkit_alone_installs_from_its_directory() -> None:
    lines = dry_run(TOOLKIT_ROOT, "install-dev", f"BIN={TOOLS}")

    assert lines == [f'{TOOLS}python -m pip install -e ".[dev]"']


def test_check_at_the_root_runs_the_toolkits_gates_then_the_use_cases_tests() -> None:
    from_root = dry_run(REPO_ROOT, "check", f"BIN={TOOLS}")[1:]
    from_toolkit = dry_run(TOOLKIT_ROOT, "check", f"BIN={TOOLS}")

    assert from_root[: len(from_toolkit)] == from_toolkit
    assert from_toolkit[0] == f"{TOOLS}ruff check src tests locustfile.py"
    assert from_toolkit[-1] == f"{TOOLS}pytest"
    # Linted with the toolkit's configuration, run from the use case's directory.
    config = f"--config ../{TOOLKIT_ROOT.name}/pyproject.toml"
    assert from_root[len(from_toolkit) :] == [
        f"cd {USE_CASE} && {TOOLS}ruff check {config} tests",
        f"cd {USE_CASE} && {TOOLS}black --check {config} tests",
        f"cd {USE_CASE} && {TOOLS}pytest",
    ]


def test_the_toolkits_makefile_keeps_only_what_needs_no_use_case() -> None:
    # The run targets read a use case's files or write its output: the use case's.
    lines = (TOOLKIT_ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    targets = {line.split(":")[0] for line in lines if re.match(r"^[\w-]+:", line)}

    assert targets == {
        "help",
        "install",
        "install-dev",
        "check",
        "format",
        "fake-domino",
        *GATES,
    }
