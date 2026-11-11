"""What the emulation's tests share: building it, credentials, a fake App behind
the proxy, a two-tier placement and named fakes for the launcher's ports."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
from fake_domino_support import (
    APPS_PATH,
    CREATE_BODY,
    INSTANCE_PATH,
    client_of,
    create,
    statuses_seen,
)
from fastapi.testclient import TestClient

from fake_domino.emulation.identity import FAKE_API_KEY, FAKE_RUN_TOKEN
from fake_domino.emulation.launcher import (
    AppLauncher,
    AppLaunchError,
    AppRunState,
    CommandError,
    CommandResult,
    CommandRunner,
    PortProbe,
)
from fake_domino.emulation.placement import AppPlacer
from fake_domino.emulation.placement_settings import (
    HardwareTierPool,
    PlacementSettings,
    SshSettings,
)
from fake_domino.emulation.proxy_settings import FakeDominoProxySettings
from fake_domino.emulation.server import create_domino_emulation
from fake_domino.state import FakeDominoApps

__all__ = ["APPS_PATH", "CREATE_BODY", "INSTANCE_PATH", "create", "statuses_seen"]
RUN_TOKEN_AUTH = {"Authorization": f"Bearer {FAKE_RUN_TOKEN}"}
API_KEY_AUTH = {"X-Domino-Api-Key": FAKE_API_KEY}

CPU_8 = HardwareTierPool("cpu-8", 8, 32, ("10.10.0.21", "10.10.0.22"), default=True)
CPU_16 = HardwareTierPool("cpu-16", 16, 64, ("10.10.0.31",))
PLACEMENT = PlacementSettings(
    SshSettings("ubuntu", "/home/ubuntu/.ssh/id_emulation"), 8888, (CPU_8, CPU_16)
)


class FakeAppLauncher(AppLauncher):
    """Records each start; reports the state a test sets per machine (default starting)."""

    def __init__(self) -> None:
        self.starts: list[tuple[str, str]] = []  # (machine, script)
        self.states: dict[str, AppRunState] = {}
        self.unreachable: set[str] = set()

    def start(self, machine: str, script: str) -> None:
        if machine in self.unreachable:
            raise AppLaunchError(f"ssh to {machine}: Connection timed out")
        self.starts.append((machine, script))

    def state(self, machine: str) -> AppRunState:
        return self.states.get(machine, AppRunState.STARTING)


class FakeCommandRunner(CommandRunner):
    """Records each command; answers with the results a test queued, or raises."""

    def __init__(self, *results: CommandResult | CommandError) -> None:
        self.calls: list[tuple[list[str], str | None, float]] = []
        self._results = list(results) or [CommandResult(0, "", "")]

    def run(
        self, argv: Sequence[str], stdin: str | None, timeout_seconds: float
    ) -> CommandResult:
        self.calls.append((list(argv), stdin, timeout_seconds))
        result = self._results.pop(0) if len(self._results) > 1 else self._results[0]
        if isinstance(result, CommandError):
            raise result
        return result


class FakePortProbe(PortProbe):
    """Says the ports a test opened are open; records each probe."""

    def __init__(self, *open_ports: tuple[str, int]) -> None:
        self.open_ports = set(open_ports)
        self.probes: list[tuple[str, int, float]] = []

    def is_open(self, host: str, port: int, timeout_seconds: float) -> bool:
        self.probes.append((host, port, timeout_seconds))
        return (host, port) in self.open_ports


def placed_apps(
    launcher: FakeAppLauncher, project_env: dict[str, str] | None = None
) -> FakeDominoApps:
    return FakeDominoApps(scheduler=AppPlacer(PLACEMENT, launcher, project_env))


def http_to(
    apps: FakeDominoApps,
    proxy: FakeDominoProxySettings | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> TestClient:
    fake = create_domino_emulation(
        apps, proxy=proxy, upstream_transport=upstream_transport
    )
    return client_of(fake)


class FakeUpstream:
    """An App running behind the fake proxy: records each request, answers 201."""

    def __init__(self) -> None:
        self.seen: list[httpx.Request] = []

    def answer(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        headers = {"X-Upstream": "fake-app", "Content-Type": "text/plain"}
        return httpx.Response(201, headers=headers, content=b"from the app")

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.answer)
