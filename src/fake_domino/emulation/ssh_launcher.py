"""``SshAppLauncher``: runs an App as a systemd unit on its machine, over ``ssh``.

Emulation only. One unit per machine (``PlacementSettings.unit``), run as the
SSH user with its output in journald (``journalctl -u rag-app``). Starting stops
and resets the previous unit, writes the start script (sent on stdin, never on
the command line: it holds the project's secrets) to ``$HOME/<unit>.sh`` with
mode 600, and starts it with ``systemd-run``. Every ``ssh`` call is
non-interactive, has a connect timeout and a timeout of its own, and is never
retried: the fake polls the state on every GET of the App instead.

Example:
    >>> SshAppLauncher(load_placement(Path("placement.json"))).state("10.10.0.21")
    <AppRunState.RUNNING: 'running'>
"""

from __future__ import annotations

import logging
import math
import shlex

from .event_log import log_event
from .launcher import (
    AppLauncher,
    AppLaunchError,
    AppRunState,
    CommandError,
    CommandResult,
    CommandRunner,
    PortProbe,
    SubprocessCommandRunner,
    TcpPortProbe,
)
from .placement_settings import PlacementSettings

# ssh's own exit code when it could not connect or log in (ssh(1), EXIT STATUS).
SSH_FAILED = 255
ACTIVE = "active"  # what `systemctl is-active` prints for a running unit
ACTIVATING = "activating"  # ... and for one systemd is still starting


class SshAppLauncher(AppLauncher):
    """Starts and checks Apps on the emulation's machines through ``ssh``.

    Args:
        settings: The SSH account and key, the unit, the App's port, timeouts.
        runner: Runs ``ssh`` locally; None runs it as a subprocess.
        probe: Checks the App's port; None makes a real TCP connect.

    Example:
        >>> SshAppLauncher(settings, FakeCommandRunner(), FakePortProbe())
    """

    def __init__(
        self,
        settings: PlacementSettings,
        runner: CommandRunner | None = None,
        probe: PortProbe | None = None,
    ) -> None:
        self.settings = settings
        self._runner = runner or SubprocessCommandRunner()
        self._probe = probe or TcpPortProbe()

    def ssh_argv(self, machine: str, remote_command: str) -> list[str]:
        """Build the ``ssh`` command line that runs ``remote_command``.

        Args:
            machine: The machine's address.
            remote_command: What the machine's shell runs.

        Returns:
            The argument list, ``ssh`` first.
        """
        ssh = self.settings.ssh
        # ConnectTimeout takes whole seconds.
        connect = math.ceil(ssh.connect_timeout_seconds)
        options = ["BatchMode=yes", "StrictHostKeyChecking=accept-new"]
        options += ["IdentitiesOnly=yes", f"ConnectTimeout={connect}"]
        flags = [part for option in options for part in ("-o", option)]
        destination = f"{ssh.user}@{machine}"
        return ["ssh", "-i", ssh.key, *flags, destination, "--", remote_command]

    def start_command(self) -> str:
        """The remote command that replaces the machine's App by the one on stdin.

        Returns:
            One shell command line, run by the SSH user's login shell.
        """
        unit, user = shlex.quote(self.settings.unit), shlex.quote(
            self.settings.ssh.user
        )
        script = f'"$HOME"/{shlex.quote(self.settings.unit + ".sh")}'
        stop_timeout = f"TimeoutStopSec={math.ceil(self.settings.stop_timeout_seconds)}"
        # stop and reset-failed fail when there is no previous unit: hence `;`.
        return (
            f"sudo systemctl stop {unit}.service 2>/dev/null; "
            f"sudo systemctl reset-failed {unit}.service 2>/dev/null; "
            f"umask 077 && cat > {script} && "
            f"sudo systemd-run --unit={unit} --uid={user} "
            f'--working-directory="$HOME" --property={stop_timeout} bash {script}'
        )

    def state_command(self) -> str:
        """The remote command that prints the unit's state.

        Returns:
            One shell command line.
        """
        return f"systemctl is-active {shlex.quote(self.settings.unit)}.service"

    def start(self, machine: str, script: str) -> None:
        """Run ``script`` on ``machine`` as its unit, replacing the previous one.

        Args:
            machine: The machine's address.
            script: The bash start script (it may hold secrets: never logged).

        Raises:
            AppLaunchError: If ``ssh`` failed, timed out or the remote command
                did not succeed; the message carries ``ssh``'s error output.
        """
        result = self._ssh(machine, self.start_command(), stdin=script)
        if result.returncode != 0:
            raise AppLaunchError(
                f"could not start unit {self.settings.unit} on {machine}: "
                f"exit {result.returncode}: {result.stderr.strip()}"
            )

    def state(self, machine: str) -> AppRunState:
        """Say how the App on ``machine`` is doing; a failed ``ssh`` is logged.

        Only the machine can say its App failed. When ``ssh`` itself cannot
        ask (refused, timed out), the state is unknown and reads as starting:
        whoever is waiting keeps waiting until its own timeout, instead of
        giving up on an App that may be fine.

        Args:
            machine: The machine's address.

        Returns:
            Running if its unit is active and its port accepts a connection;
            starting if the unit is active without the port, still activating,
            or could not be asked; failed if the machine says anything else.
        """
        try:
            result = self._ssh(machine, self.state_command(), stdin=None)
        except AppLaunchError as exc:
            return _unknown(str(exc))
        if result.returncode == SSH_FAILED:
            return _unknown(f"ssh to {machine}: {result.stderr.strip()}")
        unit_state = result.stdout.strip()
        if unit_state == ACTIVATING:
            return AppRunState.STARTING
        if unit_state != ACTIVE:
            return AppRunState.FAILED
        port, timeout = self.settings.app_port, self.settings.probe_timeout_seconds
        if self._probe.is_open(machine, port, timeout):
            return AppRunState.RUNNING
        return AppRunState.STARTING

    def _ssh(self, machine: str, command: str, stdin: str | None) -> CommandResult:
        # A local failure (ssh missing, timed out) reads like a remote one.
        argv = self.ssh_argv(machine, command)
        timeout = self.settings.ssh.command_timeout_seconds
        try:
            return self._runner.run(argv, stdin=stdin, timeout_seconds=timeout)
        except CommandError as exc:
            raise AppLaunchError(f"ssh to {machine}: {exc}") from exc


def _unknown(error: str) -> AppRunState:
    # ssh could not ask the machine: log why and report the App as still starting.
    log_event(logging.WARNING, event="app_state_unknown", error=error)
    return AppRunState.STARTING
