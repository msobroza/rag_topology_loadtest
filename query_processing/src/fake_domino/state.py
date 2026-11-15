"""The fake Domino's state: its Apps and versions, and the scripted behaviours.

What a new version's instance reports is decided by an ``BaseInstanceScheduler``
(``instances.py``): by default scripted, one status per GET of the App (a
behaviour), so a test or a dry run can watch ``rag-deploy`` wait. The routes in
``apps_routes.py`` and ``stand_ins.py`` read and change this state, all of it in
memory.

Example:
    >>> FakeDominoApps.from_behaviour("slow")  # Queued ... then Running
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from .api_shapes import (
    AppCreationRequest,
    AppInstanceSummaryResponse,
    AppVersionCreationRequest,
    AppVersionResponse,
)
from .instances import (
    BaseInstanceScheduler,
    FakeApp,
    FakeVersion,
    ScriptedInstances,
    VersionRefusedError,
    instance_id,
)

FAKE_OWNER = "fake-owner"  # the fake's one user, who owns every App
FAKE_PROJECT_NAME = "fake-project"  # the name of every App's project

# --- Behaviours: what a new version's instance reports, one status per GET ---


class FakeBehaviour(NamedTuple):
    """A named script of what a new version's instance reports, per GET."""

    description: str
    statuses: tuple[str | None, ...]  # None = no instance; the last one repeats
    forbidden: bool = False


FAKE_DOMINO_BEHAVIOURS: dict[str, FakeBehaviour] = {
    "starts": FakeBehaviour(
        "a new version's instance is Pending on the first GET, then Running",
        ("Pending", "Running"),
    ),
    "slow": FakeBehaviour(
        "Queued, Pending, Preparing, Pulling, then Running",
        ("Queued", "Pending", "Preparing", "Pulling", "Running"),
    ),
    "fails": FakeBehaviour("Pending, then Failed", ("Pending", "Failed")),
    "never-starts": FakeBehaviour(
        "Domino creates the App or version but no instance (6.2 has no start call)",
        (None,),
    ),
    "forbidden": FakeBehaviour(
        "every Apps API call answers 403 with a FailureEnvelopeV1", (None,), True
    ),
}


class FakeDominoApps:
    """The fake's state: its Apps, what their instances report, what it received.

    Args:
        statuses: What a new version's instance reports, one per GET of the App
            (the last repeats; None = no instance).
        stale_gets: GETs after a new version that still report the previous one
            as current and Running.
        forbidden: Whether every Apps API call answers 403.
        scheduler: Decides what instances report instead of ``statuses`` and
            ``stale_gets``; None scripts them (``ScriptedInstances``).

    Example:
        >>> create_fake_domino(FakeDominoApps(["Pending", "Running"], stale_gets=2))
    """

    def __init__(
        self,
        statuses: Sequence[str | None] = ("Pending", "Running"),
        *,
        stale_gets: int = 0,
        forbidden: bool = False,
        scheduler: BaseInstanceScheduler | None = None,
    ) -> None:
        self.statuses, self.forbidden = list(statuses), forbidden
        self.scheduler = scheduler or ScriptedInstances(statuses, stale_gets)
        self.apps: dict[str, FakeApp] = {}
        self.received: list[tuple[str, str]] = []  # (method, path), in order
        self.creates: list[AppCreationRequest] = []
        self.new_versions: list[tuple[str, AppVersionCreationRequest]] = []

    @classmethod
    def from_behaviour(
        cls, name: str, *, scheduler: BaseInstanceScheduler | None = None
    ) -> FakeDominoApps:
        """Build the state for one of ``FAKE_DOMINO_BEHAVIOURS``.

        Args:
            name: A key of ``FAKE_DOMINO_BEHAVIOURS``, e.g. ``"fails"``.
            scheduler: Replaces the behaviour's statuses; only ``forbidden``
                then changes anything.

        Returns:
            A fresh state scripted with that behaviour's statuses.

        Raises:
            ValueError: If ``name`` is not a known behaviour.
        """
        if name not in FAKE_DOMINO_BEHAVIOURS:
            raise ValueError(
                f"unknown fake Domino behaviour {name!r}; expected one of: "
                f"{', '.join(FAKE_DOMINO_BEHAVIOURS)}"
            )
        behaviour = FAKE_DOMINO_BEHAVIOURS[name]
        forbidden = behaviour.forbidden
        return cls(behaviour.statuses, forbidden=forbidden, scheduler=scheduler)

    def add_app(self, request: AppCreationRequest) -> FakeApp:
        """Store an App with its first version, e.g. to set up a test.

        Args:
            request: The create request; its ``version`` becomes the first version.

        Returns:
            The stored App, with id ``app-<n>``.

        Raises:
            VersionRefusedError: If the scheduler refuses the version; the App
                is then not stored.
        """
        app_id = f"app-{len(self.apps) + 1}"
        version = FakeVersion(f"{app_id}-v1", request.version)
        app = self.apps[app_id] = FakeApp(app_id, request, [version])
        try:
            self.scheduler.version_added(app)
        except VersionRefusedError:
            del self.apps[app_id]
            raise
        return app

    def add_version(self, app: FakeApp, request: AppVersionCreationRequest) -> None:
        """Append a version to ``app``; it becomes the current one.

        Args:
            app: The App that gets the version.
            request: The new version's create request.

        Raises:
            VersionRefusedError: If the scheduler refuses the version; it is
                then not stored.
        """
        app.versions.append(FakeVersion(f"{app.id}-v{len(app.versions) + 1}", request))
        try:
            self.scheduler.version_added(app)
        except VersionRefusedError:
            app.versions.pop()
            raise

    def current(self, app: FakeApp) -> AppVersionResponse:
        """Show ``app``'s current version as a list or a create shows it (not a GET).

        Args:
            app: The App whose latest version is shown.

        Returns:
            The latest version, its instance at the status the next GET reports.
        """
        return _version_response(*self.scheduler.shown(app))

    def poll(self, app: FakeApp) -> AppVersionResponse:
        """Report what one GET of ``app`` shows as its current version.

        Args:
            app: The App being fetched.

        Returns:
            What the scheduler shows: scripted, the previous version as Running
            while stale GETs remain, otherwise the latest moved one status along.
        """
        return _version_response(*self.scheduler.polled(app))

    def instances(
        self, app: FakeApp, version_id: str
    ) -> list[AppInstanceSummaryResponse] | None:
        """List a version's instances, as its last GET showed.

        Only the latest version has one: what Domino does with the previous
        version's instance after a redeploy is not documented.

        Args:
            app: The App the version belongs to.
            version_id: The version whose instances are listed.

        Returns:
            None if ``app`` has no such version; otherwise the instance of the
            latest version once it has one, or an empty list.
        """
        if version_id not in (version.id for version in app.versions):
            return None
        version = app.versions[-1]
        status = self.scheduler.last_status(app)
        if version.id != version_id or status is None:
            return []
        return [_instance(version, status)]

    def instance_owner(self, instance: str) -> tuple[FakeApp, FakeVersion] | None:
        """Find the App and version whose instance is ``instance``.

        Args:
            instance: An instance id, e.g. ``"app-1-v1-i"``.

        Returns:
            That App and version, or None if no App has it.
        """
        owners = ((a, v) for a in self.apps.values() for v in a.versions)
        return next((o for o in owners if instance_id(o[1]) == instance), None)


_VERSION_DEFAULTS: dict[str, object] = {
    "extendedIdentityPropagationToAppsEnabled": False,
    "environmentId": "",
    "environmentRevisionId": "",
    "hardwareTierId": "",
    "externalVolumeMountIds": [],
    "netAppVolumeIds": [],
    "tags": [],
}


def _instance(version: FakeVersion, status: str) -> AppInstanceSummaryResponse:
    return AppInstanceSummaryResponse(
        id=instance_id(version), createdAt=0, status=status
    )


def _version_response(version: FakeVersion, status: str | None) -> AppVersionResponse:
    instance = _instance(version, status) if status is not None else None
    fields = version.request.model_dump(exclude_none=True)
    return AppVersionResponse.model_validate(
        {**_VERSION_DEFAULTS, **fields, "id": version.id, "createdAt": 0}
        | {"updatedAt": 0, "currentInstance": instance}
    )
