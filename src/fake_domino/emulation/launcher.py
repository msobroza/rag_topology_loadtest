"""The ports the emulation starts Apps through, and their real implementations.

Emulation only: ``AppLauncher`` runs an App's start script on a machine and
says how it is doing; ``SshAppLauncher`` (``ssh_launcher.py``) implements it
over two smaller ports, ``CommandRunner`` (a local command, by default a
subprocess) and ``PortProbe`` (a TCP connect), so tests never spawn ``ssh`` or
open a socket.

Example:
    >>> SubprocessCommandRunner().run(["true"], stdin=None, timeout_seconds=5)
    CommandResult(returncode=0, stdout='', stderr='')
"""

from __future__ import annotations

import abc
import socket
import subprocess
from collections.abc import Sequence
from enum import StrEnum
from typing import NamedTuple


class AppRunState(StrEnum):
    """How an App's process on its machine is doing."""

    STARTING = "starting"  # its unit is active, its port not open yet
    RUNNING = "running"  # its unit is active and its port answers
    FAILED = "failed"  # its unit is not active, or the machine did not answer


class AppLaunchError(RuntimeError):
    """An App's start script could not be started on its machine."""


class AppLauncher(abc.ABC):
    r"""Runs Apps on machines, one App per machine.

    Example:
        >>> launcher.start("10.10.0.21", "export A=1\nexec bash app.sh\n")
    """

    @abc.abstractmethod
    def start(self, machine: str, script: str) -> None:
        """Run ``script`` on ``machine``, replacing whatever App ran there.

        Args:
            machine: The machine's address.
            script: The bash start script (it may hold secrets: never log it).

        Raises:
            AppLaunchError: If the script could not be started there.
        """

    @abc.abstractmethod
    def state(self, machine: str) -> AppRunState:
        """Say how the App on ``machine`` is doing.

        Args:
            machine: The machine's address.

        Returns:
            Starting, running or failed.
        """


class CommandResult(NamedTuple):
    """What a finished local command left behind."""

    returncode: int
    stdout: str
    stderr: str


class CommandError(RuntimeError):
    """A local command could not be run, or did not finish in time."""


class CommandRunner(abc.ABC):
    """Runs a local command to its end.

    Example:
        >>> runner.run(["ssh", "host", "--", "true"], stdin=None, timeout_seconds=20)
    """

    @abc.abstractmethod
    def run(
        self, argv: Sequence[str], stdin: str | None, timeout_seconds: float
    ) -> CommandResult:
        """Run ``argv`` with ``stdin`` as its input.

        Args:
            argv: The program and its arguments.
            stdin: Text written to its standard input; None gives it none.
            timeout_seconds: How long it may run before it is killed.

        Returns:
            Its exit code and output.

        Raises:
            CommandError: If it cannot be started or does not finish in time.
        """


class SubprocessCommandRunner(CommandRunner):
    """Runs a command as a subprocess, its output captured as text.

    Example:
        >>> SubprocessCommandRunner().run(["true"], stdin=None, timeout_seconds=5)
    """

    def run(
        self, argv: Sequence[str], stdin: str | None, timeout_seconds: float
    ) -> CommandResult:
        """Run ``argv`` with ``stdin`` as its input.

        Args:
            argv: The program and its arguments.
            stdin: Text written to its standard input; None gives it none.
            timeout_seconds: How long it may run before it is killed.

        Returns:
            Its exit code and output.

        Raises:
            CommandError: If it cannot be started or does not finish in time.
        """
        try:
            done = subprocess.run(
                list(argv),
                input=stdin,
                stdin=None if stdin is not None else subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CommandError(f"{argv[0]} timed out after {exc.timeout} s") from exc
        except OSError as exc:
            raise CommandError(f"{argv[0]} could not be started: {exc}") from exc
        return CommandResult(done.returncode, done.stdout, done.stderr)


class PortProbe(abc.ABC):
    """Says whether a TCP port accepts a connection."""

    @abc.abstractmethod
    def is_open(self, host: str, port: int, timeout_seconds: float) -> bool:
        """Try one TCP connect to ``host:port``.

        Args:
            host: The machine's address.
            port: The port.
            timeout_seconds: How long the connect may take.

        Returns:
            Whether the connection was accepted.
        """


class TcpPortProbe(PortProbe):
    """Probes a port with a real TCP connect, closed at once.

    Example:
        >>> TcpPortProbe().is_open("10.10.0.21", 8888, timeout_seconds=1)
    """

    def is_open(self, host: str, port: int, timeout_seconds: float) -> bool:
        """Try one TCP connect to ``host:port``.

        Args:
            host: The machine's address.
            port: The port.
            timeout_seconds: How long the connect may take.

        Returns:
            Whether the connection was accepted.
        """
        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                return True
        except OSError:
            return False
