"""The repository's Makefile and the toolkit's: the root installs doclang's core
from its clone, the core, the toolkit and the use case together, or stops in one
line with no clone, runs the gates over the core and its tests,
hands them to the toolkit's, then runs them over the use case and over the
charts package, each followed by its own tests.

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
# The core's directory, beside the toolkit's: what toolkits exchange, which the
# toolkit installs. Its gates and tests run first. Named only, as the use case is.
CORE = "query_processing_core"
# The targets that run over the core before the toolkit; deadcode and test are
# the toolkit's alone.
BEFORE_THE_TOOLKIT = ("check", "format", "lint", "typecheck", "complexity", "deps")
# The RAG use case's directory, beside the toolkit's: the root installs both and
# runs the use case's tests. Named only: no test of the toolkit reads its files.
USE_CASE = "query_processing_rag_usecase"
# The charts package's directory, beside them: not the product, installed for
# development only, its gates and tests run last. Named only, as the use case is.
CHARTS = "experiment_charts"
# doclang's core, in a clone of document_doclang: private and on no index, so the
# root installs it from the clone, before the core that depends on it.
DOCLANG_CORE = "packages/core"


def make_n(
    directory: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """``make -n <args>`` run in ``directory``, with ``env`` added to this one's."""
    # A make running this suite would hand its own flags and variables down, and
    # DOCLANG_CLONE is given by the test that needs it, never by the caller's shell.
    inherited = ("MAKEFLAGS", "MFLAGS", "DOCLANG_CLONE")
    base = {k: v for k, v in os.environ.items() if k not in inherited}
    return subprocess.run(
        ["make", "-n", *args],
        cwd=directory,
        env={**base, "MAKELEVEL": "0", **(env or {})},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def dry_run(
    directory: Path, *args: str, env: dict[str, str] | None = None
) -> list[str]:
    """The commands ``make -n <args>`` prints in ``directory``, one per line."""
    run = make_n(directory, *args, env=env)
    assert run.returncode == 0, f"make -n {' '.join(args)} failed: {run.stderr}"
    # GNU make 4 says where a recursive make enters and leaves; 3.81 does not.
    return [
        line.strip()
        for line in run.stdout.splitlines()
        if not re.match(r"\S*make(\[\d+\])?: (Entering|Leaving) directory", line)
    ]


def a_clone(where: Path) -> Path:
    """A stand-in clone of document_doclang at ``where``: its core's manifest alone."""
    (where / DOCLANG_CORE).mkdir(parents=True)
    (where / DOCLANG_CORE / "pyproject.toml").write_text("", encoding="utf-8")
    return where


def main_checkout() -> Path:
    """The repository's main checkout, which a worktree shares its git directory with."""
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return Path(common.stdout.strip()).parent


@pytest.mark.parametrize("target", ["check", "format", *GATES])
def test_the_root_hands_each_gate_to_the_toolkit_with_the_tools_place(
    target: str,
) -> None:
    lines = dry_run(REPO_ROOT, target, f"BIN={TOOLS}")
    core = [line for line in lines if line.startswith(f"cd {CORE} && ")]

    first_core = len(core) if target in BEFORE_THE_TOOLKIT else 0
    assert lines[:first_core] == core
    toolkit_line = lines[first_core]
    assert toolkit_line.endswith(f' -C {TOOLKIT_ROOT.name} {target} BIN="{TOOLS}"')


INSTALLS = ("install", "install-dev")


@pytest.mark.parametrize(
    ("target", "toolkit", "for_development"),
    [
        ("install", "./query_processing", ""),
        ("install-dev", '"./query_processing[dev]"', f" -e ./{CHARTS}"),
    ],
)
@pytest.mark.parametrize("given", ["on-the-command-line", "in-the-environment"])
def test_the_root_installs_doclang_the_core_the_toolkit_and_the_use_case_in_one_command(
    target: str, toolkit: str, for_development: str, given: str, tmp_path: Path
) -> None:
    # No package index has doclang's core, the core or the toolkit: the core's
    # dependency on doclang's, the toolkit's on the core and the use case's on
    # the toolkit are met only when the directories are named, in that order.
    # The charts package is no part of what runs: only install-dev adds it.
    clone = a_clone(tmp_path / "elsewhere")
    naming = f"DOCLANG_CLONE={clone}"
    on_the_command_line = given == "on-the-command-line"
    lines = dry_run(
        REPO_ROOT,
        target,
        f"BIN={TOOLS}",
        *([naming] if on_the_command_line else []),
        env=None if on_the_command_line else {"DOCLANG_CLONE": str(clone)},
    )

    doclang = f"-e {clone}/{DOCLANG_CORE}"
    installed = f"-e ./{CORE} -e {toolkit} -e ./{USE_CASE}{for_development}"
    assert lines == [f"{TOOLS}python -m pip install {doclang} {installed}"]
    assert (REPO_ROOT / toolkit.strip('"').removesuffix("[dev]")).resolve() == (
        TOOLKIT_ROOT
    )


@pytest.mark.parametrize("target", INSTALLS)
def test_the_clone_is_found_beside_the_main_checkout(target: str) -> None:
    # From git, so that a worktree, wherever it sits, finds the clone beside
    # the checkout it was made from; CI clones it beside its checkout too.
    lines = dry_run(REPO_ROOT, target, f"BIN={TOOLS}")

    clone = main_checkout().parent / "document_doclang"
    assert lines[0].startswith(
        f"{TOOLS}python -m pip install -e {clone}/{DOCLANG_CORE} "
    )


@pytest.mark.parametrize("target", INSTALLS)
def test_with_no_clone_the_install_stops_in_one_line(
    target: str, tmp_path: Path
) -> None:
    nowhere = tmp_path / "no_clone_here"

    run = make_n(REPO_ROOT, target, f"BIN={TOOLS}", f"DOCLANG_CLONE={nowhere}")

    assert run.returncode != 0
    assert run.stdout == ""  # nothing installed
    [line] = run.stderr.splitlines()
    assert "DOCLANG_CLONE" in line
    assert f"no clone of document_doclang in {nowhere}" in line


def test_the_toolkit_alone_installs_from_its_directory() -> None:
    lines = dry_run(TOOLKIT_ROOT, "install-dev", f"BIN={TOOLS}")

    assert lines == [f'{TOOLS}python -m pip install -e ../{CORE} -e ".[dev]"']


def gates_beside_the_toolkit(directory: str) -> list[str]:
    """What ``check`` runs over a distribution beside the toolkit, from its directory.

    Its source and tests, linted and type-checked with the toolkit's
    configuration, then its own tests.
    """
    pyproject = f"../{TOOLKIT_ROOT.name}/pyproject.toml"
    config = f"--config {pyproject}"
    return [
        f"cd {directory} && {TOOLS}ruff check {config} src tests",
        f"cd {directory} && {TOOLS}black --check {config} src tests",
        f"cd {directory} && {TOOLS}pydoclint {config} src",
        f"cd {directory} && {TOOLS}mypy --config-file {pyproject} src",
        f"cd {directory} && {TOOLS}ty check --python {TOOLS}python src",
        f"cd {directory} && {TOOLS}complexipy src --max-complexity-allowed 10 --failed",
        f"cd {directory} && {TOOLS}deptry src",
        f"cd {directory} && {TOOLS}pytest",
    ]


def test_check_at_the_root_runs_the_toolkits_gates_then_those_of_the_two_beside() -> (
    None
):
    core = gates_beside_the_toolkit(CORE)
    from_root = dry_run(REPO_ROOT, "check", f"BIN={TOOLS}")
    from_toolkit = dry_run(TOOLKIT_ROOT, "check", f"BIN={TOOLS}")

    # The core first, since the toolkit installs it; then the toolkit's own.
    assert from_root[: len(core)] == core
    from_root = from_root[len(core) + 1 :]
    assert from_root[: len(from_toolkit)] == from_toolkit
    assert from_toolkit[0] == f"{TOOLS}ruff check src tests locustfile.py"
    assert from_toolkit[-1] == f"{TOOLS}pytest"
    # The use case first, since it is the product's; the charts package last.
    assert from_root[len(from_toolkit) :] == [
        *gates_beside_the_toolkit(USE_CASE),
        *gates_beside_the_toolkit(CHARTS),
    ]


@pytest.mark.parametrize("gate", ["lint", "typecheck", "complexity", "deps"])
def test_a_gate_run_from_the_root_also_runs_over_the_two_beside(gate: str) -> None:
    lines = dry_run(REPO_ROOT, gate, f"BIN={TOOLS}")

    beside = [line.split(" && ")[0] for line in lines if line.startswith("cd ")]
    assert list(dict.fromkeys(beside)) == [
        f"cd {CORE}",
        f"cd {USE_CASE}",
        f"cd {CHARTS}",
    ]
    assert all(f" && {TOOLS}" in line for line in lines if line.startswith("cd "))


@pytest.mark.parametrize(
    ("tools", "told"),
    [(TOOLS, f" --python {TOOLS}python"), ("", "")],
    ids=["tools-of-a-virtualenv", "tools-on-the-path"],
)
def test_ty_beside_the_toolkit_is_told_the_python_its_tools_come_from(
    tools: str, told: str
) -> None:
    # ty checks imports against the first python on the PATH unless it is told
    # of one. Called by its path in the repository's .venv on a machine whose
    # PATH leads with another python (the CI's image), it could not find the
    # toolkit the use case imports: the first run of pull request 29. With the
    # tools on the PATH it is told nothing, and finds the active environment.
    lines = dry_run(REPO_ROOT, "typecheck", f"BIN={tools}")

    beside = [line for line in lines if line.startswith("cd ") and "ty " in line]
    assert beside == [
        f"cd {directory} && {tools}ty check{told} src"
        for directory in (CORE, USE_CASE, CHARTS)
    ]


def test_the_charts_are_redrawn_from_the_root() -> None:
    # The command reads experiments/ and writes experiments-explained/ by those
    # names, relative to where it runs: both folders are the root's.
    lines = dry_run(REPO_ROOT, "experiment-charts", f"BIN={TOOLS}")

    assert lines == [f"{TOOLS}python -m {CHARTS}"]
    assert (REPO_ROOT / "experiments").is_dir()
    assert (REPO_ROOT / "experiments-explained").is_dir()


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
        "pin-answers",
        "fake-domino",
        *GATES,
    }


def test_pin_answers_runs_the_pinned_record_test_to_make_the_record_again() -> None:
    # The test that compares the app with the pinned answer record, which the
    # option turns into the run that writes it.
    lines = dry_run(TOOLKIT_ROOT, "pin-answers", f"BIN={TOOLS}")

    test_file = "tests/testing/test_pinned_answer_record.py"
    assert lines == [f"{TOOLS}pytest {test_file} --pin-answers"]
    assert (TOOLKIT_ROOT / test_file).is_file()
