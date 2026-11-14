"""``SshAppLauncher``: the exact ``ssh`` commands it runs, and how it reads their answer."""

from __future__ import annotations

import logging

import pytest
from test_fake_domino_emulation_support import (
    PLACEMENT,
    FakeCommandRunner,
    FakePortProbe,
)

from fake_domino.emulation.launcher import (
    AppLaunchError,
    AppRunState,
    CommandError,
    CommandResult,
    SubprocessCommandRunner,
)
from fake_domino.emulation.ssh_launcher import SshAppLauncher

MACHINE = "10.10.0.21"
SSH = [
    "ssh",
    "-i",
    "/home/ubuntu/.ssh/id_emulation",
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "IdentitiesOnly=yes",
    "-o",
    "ConnectTimeout=5",
    "ubuntu@10.10.0.21",
    "--",
]
START = (
    "sudo systemctl stop rag-app.service 2>/dev/null; "
    "sudo systemctl reset-failed rag-app.service 2>/dev/null; "
    'umask 077 && cat > "$HOME"/rag-app.sh && '
    "sudo systemd-run --unit=rag-app --uid=ubuntu "
    '--working-directory="$HOME" --property=TimeoutStopSec=10 bash "$HOME"/rag-app.sh'
)
SCRIPT = "export RAG_ES_PASSWORD='s3cret'\nexec bash domino/app.sh\n"


def launcher(
    runner: FakeCommandRunner, probe: FakePortProbe | None = None
) -> SshAppLauncher:
    return SshAppLauncher(PLACEMENT, runner, probe or FakePortProbe())


def test_start_sends_the_script_on_stdin_and_never_on_the_command_line() -> None:
    runner = FakeCommandRunner()
    launcher(runner).start(MACHINE, SCRIPT)
    assert runner.calls == [([*SSH, START], SCRIPT, 20.0)]


def test_a_start_that_ssh_cannot_complete_raises_with_its_error_output() -> None:
    runner = FakeCommandRunner(
        CommandResult(255, "", "ssh: connect to host: timed out\n")
    )
    with pytest.raises(AppLaunchError) as caught:
        launcher(runner).start(MACHINE, SCRIPT)
    assert str(caught.value) == (
        "could not start unit rag-app on 10.10.0.21: exit 255: "
        "ssh: connect to host: timed out"
    )
    assert "s3cret" not in str(caught.value)


def test_a_start_whose_ssh_times_out_raises_naming_the_machine() -> None:
    runner = FakeCommandRunner(CommandError("ssh timed out after 20.0 s"))
    with pytest.raises(AppLaunchError, match=r"^ssh to 10.10.0.21: ssh timed out"):
        launcher(runner).start(MACHINE, SCRIPT)


@pytest.mark.parametrize(
    ("printed", "port_open", "state"),
    [
        ("active\n", True, AppRunState.RUNNING),
        ("active\n", False, AppRunState.STARTING),
        ("failed\n", True, AppRunState.FAILED),
        ("inactive\n", False, AppRunState.FAILED),
        ("activating\n", False, AppRunState.STARTING),
    ],
)
def test_state_reads_the_unit_then_probes_the_apps_port(
    printed: str, port_open: bool, state: AppRunState
) -> None:
    runner = FakeCommandRunner(
        CommandResult(0 if printed == "active\n" else 3, printed, "")
    )
    probe = FakePortProbe(*([(MACHINE, 8888)] if port_open else []))
    assert launcher(runner, probe).state(MACHINE) is state
    assert runner.calls == [([*SSH, "systemctl is-active rag-app.service"], None, 20.0)]
    expected_probes = [(MACHINE, 8888, 1.0)] if printed == "active\n" else []
    assert probe.probes == expected_probes


@pytest.mark.parametrize(
    "answer",
    [CommandResult(255, "", "ssh: Connection refused"), CommandError("ssh timed out")],
)
def test_a_state_ssh_cannot_get_reads_as_still_starting_and_is_logged(
    answer: CommandResult | CommandError, caplog: pytest.LogCaptureFixture
) -> None:
    # Only the machine can say its App failed: an ssh hiccup must not make
    # rag-deploy give up on an App that may be fine.
    caplog.set_level(logging.WARNING, logger="fake_domino")
    probe = FakePortProbe((MACHINE, 8888))
    state = launcher(FakeCommandRunner(answer), probe).state(MACHINE)

    assert state is AppRunState.STARTING
    assert probe.probes == []
    [record] = caplog.records
    assert '"event": "app_state_unknown"' in record.message
    assert "10.10.0.21" in record.message


def test_the_subprocess_runner_feeds_stdin_and_reports_the_exit() -> None:
    # A local `cat`, not ssh: the default runner itself is real.
    result = SubprocessCommandRunner().run(["cat"], stdin="hello", timeout_seconds=5)
    assert result == CommandResult(0, "hello", "")


def test_the_subprocess_runner_turns_a_timeout_into_a_command_error() -> None:
    with pytest.raises(CommandError, match=r"^sleep timed out after 0\.1 s$"):
        SubprocessCommandRunner().run(["sleep", "5"], stdin=None, timeout_seconds=0.1)


def test_the_subprocess_runner_turns_a_missing_program_into_a_command_error() -> None:
    with pytest.raises(CommandError, match=r"^no-such-program could not be started"):
        SubprocessCommandRunner().run(["no-such-program"], None, timeout_seconds=5)
