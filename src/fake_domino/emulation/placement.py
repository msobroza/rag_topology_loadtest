"""Placing Apps on the emulation's machines: the stand-in for Domino's scheduler.

Emulation only. A new App takes a free machine of its hardware tier and keeps it
for the fake's lifetime (the 6.2 API has no stop call to imitate); a new version
of the App is started on that same machine. With no free machine the App's
instance is ``Queued`` and each later GET of the App tries again. Once started,
the instance's status is the launcher's: starting is ``Preparing``, running
``Running``, failed ``Failed``.

Assumed (Domino's answers are undocumented): an unknown ``hardwareTierId`` is
refused with 400, and a new version asking for another tier than the machine its
App holds with 409, both as a ``FailureEnvelopeV1``.

``AppPlacer`` is the core's ``InstanceScheduler`` seam, plugged into
``FakeDominoApps(scheduler=...)``.

Example:
    >>> placer = AppPlacer(settings, SshAppLauncher(settings), project_env={})
    >>> FakeDominoApps(scheduler=placer)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import NamedTuple

from ..instances import (
    FakeApp,
    FakeVersion,
    InstanceScheduler,
    InstanceView,
    VersionRefusedError,
    instance_id,
)
from ..state import FAKE_OWNER, FAKE_PROJECT_NAME
from .event_log import log_event
from .launcher import AppLauncher, AppLaunchError, AppRunState
from .placement_settings import HardwareTierPool, PlacementSettings
from .start_script import DominoRunVariables, compose_start_script

QUEUED, PREPARING, RUNNING, FAILED = "Queued", "Preparing", "Running", "Failed"
_STATUS_OF = {
    AppRunState.STARTING: PREPARING,
    AppRunState.RUNNING: RUNNING,
    AppRunState.FAILED: FAILED,
}


class PlacementRefusedError(VersionRefusedError):
    """An App version the emulation cannot place, with the HTTP status to answer."""


class AppLaunch(NamedTuple):
    """One App version to start: its run variables and its ``entryScript``."""

    run: DominoRunVariables
    entry_script: str


@dataclass
class _Placement:
    # One App's machine (None while Queued), its latest version and last status.
    tier: HardwareTierPool
    launch: AppLaunch
    machine: str | None = None
    status: str = QUEUED
    start_failed: bool = False


class AppPlacer(InstanceScheduler):
    """Holds which App runs on which machine, and starts and checks them.

    Args:
        settings: The tiers and their machines.
        launcher: Starts an App on a machine and reports its state.
        project_env: The project environment variables every App gets.

    Example:
        >>> AppPlacer(settings, FakeAppLauncher(), project_env={"K": "v"})
    """

    def __init__(
        self,
        settings: PlacementSettings,
        launcher: AppLauncher,
        project_env: Mapping[str, str] | None = None,
    ) -> None:
        self.settings = settings
        self._launcher = launcher
        self._project_env = dict(project_env or {})
        self._placements: dict[str, _Placement] = {}
        # Routes run in a thread pool: two Apps must not take the same machine.
        self._lock = threading.Lock()

    def version_added(self, app: FakeApp) -> None:
        """Start ``app``'s new version: on its machine, on a free one, or Queued.

        Args:
            app: The App; its last version is the new one.

        Raises:
            PlacementRefusedError: If its tier is unknown or not its machine's.
        """
        version = app.versions[-1]
        tier = self.tier_for(app.id, version.request.hardwareTierId)
        run = DominoRunVariables(
            hardware_tier_id=tier.id,
            project_id=app.request.projectId,
            app_id=app.id,
            app_version_id=version.id,
            run_host_path=run_host_path(version),
        )
        self.launch(app.id, AppLaunch(run, version.request.entryScript or ""))

    def shown(self, app: FakeApp) -> InstanceView:
        """The latest version, at the status its last GET left.

        Args:
            app: The App.

        Returns:
            Its latest version and that status.
        """
        return InstanceView(app.versions[-1], self.status(app.id))

    def polled(self, app: FakeApp) -> InstanceView:
        """The latest version, at the status its machine reports now.

        Args:
            app: The App being fetched.

        Returns:
            Its latest version and that status.
        """
        return InstanceView(app.versions[-1], self.refresh(app.id))

    def last_status(self, app: FakeApp) -> str | None:
        """The latest version's instance status, as the last GET left it.

        Args:
            app: The App.

        Returns:
            Queued, Preparing, Running or Failed.
        """
        return self.status(app.id)

    def tier_for(self, app_id: str, tier_id: str | None) -> HardwareTierPool:
        """Resolve the tier a version of ``app_id`` asks for, before it is stored.

        Args:
            app_id: The App the version is for (new or existing).
            tier_id: The version's ``hardwareTierId``; empty means the default.

        Returns:
            The tier.

        Raises:
            PlacementRefusedError: 400 if no tier has that id; 409 if the App
                already holds a machine of another tier.
        """
        tier = self.settings.tier(tier_id)
        if tier is None:
            known = [tier.id for tier in self.settings.tiers]
            raise PlacementRefusedError(
                HTTPStatus.BAD_REQUEST,
                f"unknown hardwareTierId {tier_id!r}; the emulation has {known}",
            )
        held = self._placements.get(app_id)
        if held is not None and held.machine is not None and held.tier != tier:
            raise PlacementRefusedError(
                HTTPStatus.CONFLICT,
                f"App {app_id} runs on a {held.tier.id!r} machine; the emulation "
                f"cannot move it to {tier.id!r} (restart the fake Domino)",
            )
        return tier

    def launch(self, app_id: str, launch: AppLaunch) -> None:
        """Start a new version of ``app_id``: on its machine, or on a free one.

        Args:
            app_id: The App.
            launch: The version's run variables (tier resolved) and script.
        """
        tier = self.tier_for(app_id, launch.run.hardware_tier_id)
        placement = self._placements.setdefault(app_id, _Placement(tier, launch))
        # A Queued App may change tier; a placed one cannot (tier_for refuses).
        placement.tier, placement.launch, placement.start_failed = tier, launch, False
        if placement.machine is not None:
            self._start(app_id, placement, placement.machine)
            return
        self._place(app_id, placement)

    def refresh(self, app_id: str) -> str:
        """Update and report ``app_id``'s instance status, as a GET of it does.

        A Queued App tries again to take a free machine.

        Args:
            app_id: A launched App.

        Returns:
            Queued, Preparing, Running or Failed.
        """
        placement = self._placements[app_id]
        if placement.machine is None:
            self._place(app_id, placement)
        elif not placement.start_failed:
            # A failed start stays Failed: the machine may still run the old App.
            state = self._launcher.state(placement.machine)
            placement.status = _STATUS_OF[state]
        return placement.status

    def status(self, app_id: str) -> str:
        """Report ``app_id``'s instance status as its last GET left it.

        Args:
            app_id: A launched App.

        Returns:
            Queued, Preparing, Running or Failed.
        """
        return self._placements[app_id].status

    def upstream(self, app_id: str) -> str | None:
        """Where the fake's App proxy reaches ``app_id``.

        Args:
            app_id: An App, launched or not.

        Returns:
            ``http://<machine>:<app_port>``, or None while it has no machine.
        """
        placement = self._placements.get(app_id)
        if placement is None or placement.machine is None:
            return None
        return f"http://{placement.machine}:{self.settings.app_port}"

    def _place(self, app_id: str, placement: _Placement) -> None:
        # Take a free machine of the tier and start there, or stay Queued.
        with self._lock:
            held = {p.machine for p in self._placements.values()}
            free = [m for m in placement.tier.machines if m not in held]
            # Only the thread that takes the machine starts the App on it.
            taken = free[0] if placement.machine is None and free else None
            placement.machine = placement.machine or taken
        if placement.machine is None:
            placement.status = QUEUED
        if taken is None:
            return
        log_event(event="app_placed", app_id=app_id, machine=taken)
        self._start(app_id, placement, taken)

    def _start(self, app_id: str, placement: _Placement, machine: str) -> None:
        # The script holds the project's secrets: only its version is logged.
        launch = placement.launch
        script = compose_start_script(
            launch.run,
            self.settings.api_proxy_url,
            self._project_env,
            launch.entry_script,
        )
        try:
            self._launcher.start(machine, script)
        except AppLaunchError as exc:
            log_event(
                logging.ERROR, event="app_start_failed", app_id=app_id, error=str(exc)
            )
            placement.start_failed, placement.status = True, FAILED
            return
        log_event(event="app_started", app_id=app_id, version=launch.run.app_version_id)
        placement.status = PREPARING


def run_host_path(version: FakeVersion) -> str:
    """The path prefix of ``version``'s instance, as ``DOMINO_RUN_HOST_PATH``.

    Args:
        version: An App version.

    Returns:
        ``/{owner}/{project}/r/notebookSession/{instance id}/``.

    Example:
        >>> run_host_path(app.versions[0])
        '/fake-owner/fake-project/r/notebookSession/app-1-v1-i/'
    """
    owner_project = f"/{FAKE_OWNER}/{FAKE_PROJECT_NAME}"
    return f"{owner_project}/r/notebookSession/{instance_id(version)}/"
