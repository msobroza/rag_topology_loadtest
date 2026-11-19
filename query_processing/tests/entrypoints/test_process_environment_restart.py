"""``qp-serve`` and ``qp-model-server`` restart once into the environment they need.

The first start finds the allocator not preloaded and OpenMP's binding not
exported, and replaces itself, on the same interpreter and arguments, with an
environment that holds them; the restarted process finds them and serves.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pytest

from query_processing.entrypoints import qp_model_server, qp_serve

Exec = tuple[str, list[str], dict[str, str]]


class RestartedError(Exception):
    """Raised where ``execve`` would have replaced the process."""


class RecordingExecve:
    """Records each ``execve`` call and stops the caller, as a real one would."""

    def __init__(self) -> None:
        self.calls: list[Exec] = []

    def __call__(
        self, path: str, argv: Sequence[str], environment: Mapping[str, str]
    ) -> None:
        self.calls.append((path, list(argv), dict(environment)))
        raise RestartedError


def stub_qp_serve(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``qp-serve``'s app and Uvicorn replaced; return what ran."""
    ran: list[str] = []
    monkeypatch.setattr(qp_serve, "create_app", lambda composition: "the app")
    monkeypatch.setattr(qp_serve.uvicorn, "run", lambda *a, **k: ran.append("ran"))
    return ran


def stub_qp_model_server(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``qp-model-server``'s server and Uvicorn replaced; return what ran."""
    ran: list[str] = []
    monkeypatch.setattr(
        qp_model_server, "create_model_server", lambda name, composition: name
    )
    monkeypatch.setattr(
        qp_model_server.uvicorn, "run", lambda *a, **k: ran.append("ran")
    )
    return ran


COMMANDS = [
    pytest.param(qp_serve, stub_qp_serve, ["--port", "9000"], id="qp-serve"),
    pytest.param(
        qp_model_server,
        stub_qp_model_server,
        ["--serve", "reranker"],
        id="qp-model-server",
    ),
]


@pytest.mark.parametrize(("command", "stub", "arguments"), COMMANDS)
def test_main_restarts_once_with_the_same_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: ModuleType,
    stub: Callable[[pytest.MonkeyPatch], list[str]],
    arguments: list[str],
) -> None:
    library = tmp_path / "libjemalloc.so.2"
    library.touch()
    monkeypatch.setenv("QP_PROCESS_MEMORY_ALLOCATOR_LIBRARY", str(library))
    monkeypatch.setenv("QP_PROCESS_OMP_PROC_BIND", "close")
    monkeypatch.delenv("OMP_PROC_BIND", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    script = f"/venv/bin/{command.__name__.rsplit('.', 1)[-1]}"
    started = [sys.executable, script, *arguments]  # the interpreter's own argv
    monkeypatch.setattr(sys, "orig_argv", started)
    execve = RecordingExecve()
    monkeypatch.setattr(os, "execve", execve)
    ran = stub(monkeypatch)

    with pytest.raises(RestartedError):
        command.main(arguments)
    [(path, argv, environment)] = execve.calls
    for variable in ("LD_PRELOAD", "OMP_PROC_BIND"):
        monkeypatch.setenv(variable, environment[variable])

    assert command.main(arguments) == 0
    assert (path, argv) == (sys.executable, started)
    assert (environment["LD_PRELOAD"], environment["OMP_PROC_BIND"]) == (
        str(library),
        "close",
    )
    assert len(execve.calls) == 1
    assert ran == ["ran"]


@pytest.mark.parametrize(("command", "stub", "arguments"), COMMANDS)
def test_main_ends_on_binding_given_both_ways_as_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: ModuleType,
    stub: Callable[[pytest.MonkeyPatch], list[str]],
    arguments: list[str],
) -> None:
    monkeypatch.setenv("QP_PROCESS_OMP_PROC_BIND", "close")
    monkeypatch.setenv("OMP_PROC_BIND", "spread")
    monkeypatch.setattr(os, "execve", RecordingExecve())
    ran = stub(monkeypatch)

    with pytest.raises(SystemExit) as stopped:
        command.main(arguments)

    assert stopped.value.code == 2
    assert ran == []
    assert "QP_PROCESS_OMP_PROC_BIND=close" in capsys.readouterr().err
