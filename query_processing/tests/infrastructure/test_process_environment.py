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


def test_the_restart_keeps_the_interpreter_and_the_arguments() -> None:
    execve = RecordingExecve()
    argv = ["/venv/bin/qp-serve", "--port", "9000"]

    restart_with({"OMP_PROC_BIND": "close"}, argv, execve=execve)

    assert execve.calls == [
        (sys.executable, [sys.executable, *argv], {"OMP_PROC_BIND": "close"})
    ]


def test_the_arguments_default_to_those_the_process_was_started_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["/venv/bin/qp-model-server", "--serve", "x"])
    monkeypatch.setattr(sys, "executable", "/venv/bin/python3.11")
    execve = RecordingExecve()

    restart_with({"OMP_PROC_BIND": "close"}, execve=execve)

    [(path, argv, _)] = execve.calls
    assert path == "/venv/bin/python3.11"
    assert argv == ["/venv/bin/python3.11", "/venv/bin/qp-model-server", "--serve", "x"]
