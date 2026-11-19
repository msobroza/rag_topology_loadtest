"""A process restarts itself on its own interpreter and arguments, once."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence

import pytest

from query_processing.infrastructure.process_environment import restart_with

Exec = tuple[str, list[str], dict[str, str]]


class RecordingExecve:
    """Records each ``execve`` call in place of replacing the process."""

    def __init__(self) -> None:
        self.calls: list[Exec] = []

    def __call__(
        self, path: str, argv: Sequence[str], environment: Mapping[str, str]
    ) -> None:
        self.calls.append((path, list(argv), dict(environment)))


def test_no_environment_restarts_nothing() -> None:
    execve = RecordingExecve()

    restart_with(None, ["/venv/bin/qp-serve"], execve=execve)

    assert execve.calls == []


def test_the_restart_runs_the_arguments_it_is_given_on_the_interpreter() -> None:
    execve = RecordingExecve()
    argv = ["/venv/bin/python3.11", "/venv/bin/qp-serve", "--port", "9000"]

    restart_with({"OMP_PROC_BIND": "close"}, argv, execve=execve)

    assert execve.calls == [(sys.executable, argv, {"OMP_PROC_BIND": "close"})]


@pytest.mark.parametrize(
    ("orig_argv", "argv"),
    [
        pytest.param(
            ["/venv/bin/python3.11", "/venv/bin/qp-model-server", "--serve", "x"],
            ["/venv/bin/qp-model-server", "--serve", "x"],
            id="console-script",
        ),
        # sys.argv[0] is then the module's file: run as a script, its relative
        # imports would break, and the interpreter's flags would be lost.
        pytest.param(
            ["python3", "-X", "utf8", "-m", "query_processing.entrypoints.qp_serve"],
            ["/src/query_processing/entrypoints/qp_serve.py"],
            id="python-m",
        ),
    ],
)
def test_the_restart_keeps_how_the_process_was_started(
    monkeypatch: pytest.MonkeyPatch, orig_argv: list[str], argv: list[str]
) -> None:
    monkeypatch.setattr(sys, "orig_argv", orig_argv)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "executable", "/venv/bin/python3.11")
    execve = RecordingExecve()

    restart_with({"OMP_PROC_BIND": "close"}, execve=execve)

    assert execve.calls == [
        ("/venv/bin/python3.11", orig_argv, {"OMP_PROC_BIND": "close"})
    ]
