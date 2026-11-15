"""Placing Apps on the emulation's machines: the stand-in for Domino's scheduler.

Emulation only. An App runs on as many machines (replicas) as its version's
``autoscalingSpecification.minReplicas`` when the specification is enabled, else
on one; the rest of the specification is stored and echoed, never acted on.
A new App takes that many free machines of its hardware tier, all or none, and
keeps them for the fake's lifetime (the 6.2 API has no stop call to imitate).
A new version starts on the machines its App holds: one asking for more takes
free ones for the difference, all or none; one asking for fewer stops the App on
the surplus and frees it. Short of free machines the instance is ``Queued`` (the
previous version, if any, keeps running) and each later GET of the App tries
again. Every replica runs the same start script; the App still has one instance,
as on Domino, where replicas are pods of one instance. Once started, its status
is the machines': ``Running`` when all run, ``Failed`` when one failed, else
``Preparing``.

Assumed (Domino's answers are undocumented): an unknown ``hardwareTierId``, or
more replicas than its tier has machines, is refused with 400, and a new version
asking for another tier than the machines its App holds with 409, both as a
``FailureEnvelopeV1``.

``AppPlacer`` is the core's ``BaseInstanceScheduler`` seam, plugged into
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
    BaseInstanceScheduler,
    FakeApp,
    FakeVersion,
    InstanceView,
    VersionRefusedError,
    instance_id,
)
from ..state import FAKE_OWNER, FAKE_PROJECT_NAME
from .event_log import log_event
from .launcher import AppLaunchError, AppRunState, BaseAppLauncher
from .placement_settings import HardwareTierPool, PlacementSettings
from .start_script import DominoRunVariables, compose_start_script

QUEUED, PREPARING, RUNNING, FAILED = "Queued", "Preparing", "Running", "Failed"


class PlacementRefusedError(VersionRefusedError):
    """An App version the emulation cannot place, with the HTTP status to answer."""


class AppLaunch(NamedTuple):
    """One App version to start: its run variables, ``entryScript`` and replicas."""

    run: DominoRunVariables
    entry_script: str
    replicas: int = 1


@dataclass
class _Placement:
    # One App's machines (empty while it has none), its latest version and status.
    tier: HardwareTierPool
    launch: AppLaunch
    machines: tuple[str, ...] = ()
    status: str = QUEUED
    start_failed: bool = False

    def lacking(self) -> int:
        # How many more machines the latest version needs (negative: surplus).
        return self.launch.replicas - len(self.machines)


class AppPlacer(BaseInstanceScheduler):
    """Holds which App runs on which machines, and starts and checks them.

    Args:
        settings: The tiers and their machines.
        launcher: Starts and stops an App on a machine and reports its state.
        project_env: The project environment variables every App gets.

    Example:
        >>> AppPlacer(settings, FakeAppLauncher(), project_env={"K": "v"})
    """

    def __init__(
        self,
        settings: PlacementSettings,
        launcher: BaseAppLauncher,
        project_env: Mapping[str, str] | None = None,
    ) -> None:
        self.settings = settings
        self._launcher = launcher
        self._project_env = dict(project_env or {})
        self._placements: dict[str, _Placement] = {}
        # Routes run in a thread pool: two Apps must not take the same machine.
        self._lock = threading.Lock()

    def version_added(self, app: FakeApp) -> None:
        """Start ``app``'s new version: on its machines, on free ones, or Queued.

        Args:
            app: The App; its last version is the new one.

        Raises:
            PlacementRefusedError: If its tier is unknown, too small for its
                replicas, or not its machines'.
        """
        version = app.versions[-1]
        replicas = replica_count(version)
        tier = self.tier_for(app.id, version.request.hardwareTierId, replicas)
        run = DominoRunVariables(
            hardware_tier_id=tier.id,
            project_id=app.request.projectId,
            app_id=app.id,
            app_version_id=version.id,
            run_host_path=run_host_path(version),
        )
        entry_script = version.request.entryScript or ""
        self.launch(app.id, AppLaunch(run, entry_script, replicas))

    def shown(self, app: FakeApp) -> InstanceView:
        """The latest version, at the status its last GET left.

        Args:
            app: The App.

        Returns:
            Its latest version and that status.
        """
        return InstanceView(app.versions[-1], self.status(app.id))

    def polled(self, app: FakeApp) -> InstanceView:
        """The latest version, at the status its machines report now.

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

    def tier_for(
        self, app_id: str, tier_id: str | None, replicas: int
    ) -> HardwareTierPool:
        """Resolve the tier a version of ``app_id`` asks for, before it is stored.

        Args:
            app_id: The App the version is for (new or existing).
            tier_id: The version's ``hardwareTierId``; empty means the default.
            replicas: How many machines the version runs on.

        Returns:
            The tier.

        Raises:
            PlacementRefusedError: 400 if no tier has that id or it has fewer
                machines than ``replicas``; 409 if the App already holds
                machines of another tier.
        """
        tier = self.settings.tier(tier_id)
        if tier is None:
            known = [tier.id for tier in self.settings.tiers]
            raise PlacementRefusedError(
                HTTPStatus.BAD_REQUEST,
                f"unknown hardwareTierId {tier_id!r}; the emulation has {known}",
            )
        if replicas > len(tier.machines):
            # It could never run: waiting Queued would hide the mistake.
            raise PlacementRefusedError(
                HTTPStatus.BAD_REQUEST,
                f"{replicas} replicas asked, but hardware tier {tier.id!r} has "
                f"{len(tier.machines)} machine(s)",
            )
        held = self._placements.get(app_id)
        if held is not None and held.machines and held.tier != tier:
            raise PlacementRefusedError(
                HTTPStatus.CONFLICT,
                f"App {app_id} runs on a {held.tier.id!r} machine; the emulation "
                f"cannot move it to {tier.id!r} (restart the fake Domino)",
            )
        return tier

    def launch(self, app_id: str, launch: AppLaunch) -> None:
        """Start a new version of ``app_id`` on its machines, taking or freeing some.

        Args:
            app_id: The App.
            launch: The version's run variables (tier resolved), script and
                replicas.

        Raises:
            PlacementRefusedError: As ``tier_for``.
        """
        tier = self.tier_for(app_id, launch.run.hardware_tier_id, launch.replicas)
        placement = self._placements.setdefault(app_id, _Placement(tier, launch))
        # A Queued App may change tier; a placed one cannot (tier_for refuses).
        placement.tier, placement.launch, placement.start_failed = tier, launch, False
        self._release_surplus(app_id, placement)
        if placement.lacking() > 0:
            self._place(app_id, placement)
            return
        self._start(app_id, placement)

    def refresh(self, app_id: str) -> str:
        """Update and report ``app_id``'s instance status, as a GET of it does.

        An App short of machines tries again to take free ones.

        Args:
            app_id: A launched App.

        Returns:
            Queued, Preparing, Running or Failed.
        """
        placement = self._placements[app_id]
        if placement.lacking() > 0:
            self._place(app_id, placement)
        elif not placement.start_failed:
            # A failed start stays Failed: the machine may still run the old App.
            states = [self._launcher.state(m) for m in placement.machines]
            placement.status = _instance_status(states)
        return placement.status

    def status(self, app_id: str) -> str:
        """Report ``app_id``'s instance status as its last GET left it.

        Args:
            app_id: A launched App.

        Returns:
            Queued, Preparing, Running or Failed.
        """
        return self._placements[app_id].status

    def upstreams(self, app_id: str) -> tuple[str, ...]:
        """Where the fake's App proxy reaches ``app_id``: one URL per machine.

        Args:
            app_id: An App, launched or not.

        Returns:
            ``http://<machine>:<app_port>`` for each machine it holds, in order;
            empty while it holds none.

        Example:
            >>> placer.upstreams("app-1")
            ('http://10.10.0.21:8888', 'http://10.10.0.22:8888')
        """
        placement = self._placements.get(app_id)
        machines = placement.machines if placement is not None else ()
        return tuple(f"http://{m}:{self.settings.app_port}" for m in machines)

    def _place(self, app_id: str, placement: _Placement) -> None:
        # Take the free machines it lacks and start there, or stay Queued.
        taken = self._take(placement)
        for machine in taken:
            log_event(event="app_placed", app_id=app_id, machine=machine)
        if taken:
            self._start(app_id, placement)
        elif placement.lacking() > 0:
            placement.status = QUEUED

    def _take(self, placement: _Placement) -> tuple[str, ...]:
        # All or none: a part would hold machines that another App could run on.
        # Only the thread that takes them starts the App there.
        with self._lock:
            lacking = placement.lacking()
            held = {m for p in self._placements.values() for m in p.machines}
            free = [m for m in placement.tier.machines if m not in held]
            taken = tuple(free[:lacking]) if 0 < lacking <= len(free) else ()
            placement.machines += taken
        return taken

    def _release_surplus(self, app_id: str, placement: _Placement) -> None:
        # The last machines held go: stopped first, so no App ever finds one busy.
        kept = placement.machines[: placement.launch.replicas]
        surplus = placement.machines[placement.launch.replicas :]
        for machine in surplus:
            self._launcher.stop(machine)
        with self._lock:
            placement.machines = kept
        for machine in surplus:
            log_event(event="app_released", app_id=app_id, machine=machine)

    def _start(self, app_id: str, placement: _Placement) -> None:
        # The script holds the project's secrets: only its version is logged.
        launch = placement.launch
        script = compose_start_script(
            launch.run,
            self.settings.api_proxy_url,
            self._project_env,
            launch.entry_script,
        )
        failed = False
        for machine in placement.machines:
            failed |= not self._start_on(app_id, machine, launch, script)
        placement.start_failed = failed
        placement.status = FAILED if failed else PREPARING

    def _start_on(
        self, app_id: str, machine: str, launch: AppLaunch, script: str
    ) -> bool:
        # One replica; whether it started.
        try:
            self._launcher.start(machine, script)
        except AppLaunchError as exc:
            log_event(
                logging.ERROR,
                event="app_start_failed",
                app_id=app_id,
                machine=machine,
                error=str(exc),
            )
            return False
        version = launch.run.app_version_id
        log_event(event="app_started", app_id=app_id, machine=machine, version=version)
        return True


def replica_count(version: FakeVersion) -> int:
    """How many machines ``version`` runs on: its enabled minimum replicas, else 1.

    Only ``minReplicas`` is acted on: the emulation never scales during a run.

    Args:
        version: An App version.

    Returns:
        ``autoscalingSpecification.minReplicas`` when the specification is
        enabled and the minimum is at least 1; otherwise 1.

    Example:
        >>> replica_count(app.versions[0])
        1
    """
    spec = version.request.autoscalingSpecification
    if spec is None or not spec.enabled or spec.minReplicas is None:
        return 1
    return max(spec.minReplicas, 1)


def _instance_status(states: list[AppRunState]) -> str:
    # One instance for all replicas: it runs only when every one does.
    if AppRunState.FAILED in states:
        return FAILED
    if all(state is AppRunState.RUNNING for state in states):
        return RUNNING
    return PREPARING


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
