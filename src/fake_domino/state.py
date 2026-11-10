"""The fake Domino's state: its Apps and versions, and the scripted behaviours.

What a new version's instance reports is scripted, one status per GET of the App
(a behaviour), so a test or a dry run can watch ``rag-deploy`` wait. The routes
in ``apps_routes.py`` and ``stand_ins.py`` read and change this state.

Example:
    >>> FakeDominoApps.from_behaviour("slow")  # Queued ... then Running
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from .api_shapes import (
    AppCreationRequest,
    AppInstanceSummaryResponse,
    AppVersionCreationRequest,
    AppVersionResponse,
)

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


@dataclass
class _FakeVersion:
    id: str
    request: AppVersionCreationRequest
    gets: int = 0  # GETs of the App that reported this version


@dataclass
class _FakeApp:
    id: str
    request: AppCreationRequest
    versions: list[_FakeVersion]
    stale_gets_left: int = 0


class FakeDominoApps:
    """The fake's state: its Apps, its scripted statuses and what it received.

    Args:
        statuses: What a new version's instance reports, one per GET of the App
            (the last repeats; None = no instance).
        stale_gets: GETs after a new version that still report the previous one
            as current and Running.
        forbidden: Whether every Apps API call answers 403.

    Example:
        >>> create_fake_domino(FakeDominoApps(["Pending", "Running"], stale_gets=2))
    """

    def __init__(
        self,
        statuses: Sequence[str | None] = ("Pending", "Running"),
        *,
        stale_gets: int = 0,
        forbidden: bool = False,
    ) -> None:
        self.statuses, self.stale_gets = list(statuses), stale_gets
        self.forbidden = forbidden
        self.apps: dict[str, _FakeApp] = {}
        self.received: list[tuple[str, str]] = []  # (method, path), in order
        self.creates: list[AppCreationRequest] = []
        self.new_versions: list[tuple[str, AppVersionCreationRequest]] = []

    @classmethod
    def from_behaviour(cls, name: str) -> FakeDominoApps:
        """Build the state for one of ``FAKE_DOMINO_BEHAVIOURS``.

        Args:
            name: A key of ``FAKE_DOMINO_BEHAVIOURS``, e.g. ``"fails"``.

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
        return cls(behaviour.statuses, forbidden=behaviour.forbidden)

    def add_app(self, request: AppCreationRequest) -> _FakeApp:
        """Store an App with its first version, e.g. to set up a test.

        Args:
            request: The create request; its ``version`` becomes the first version.

        Returns:
            The stored App, with id ``app-<n>``.
        """
        app_id = f"app-{len(self.apps) + 1}"
        version = _FakeVersion(f"{app_id}-v1", request.version)
        self.apps[app_id] = _FakeApp(app_id, request, [version])
        return self.apps[app_id]

    def add_version(self, app: _FakeApp, request: AppVersionCreationRequest) -> None:
        """Append a version to ``app``, which then reports it after ``stale_gets``.

        Args:
            app: The App that gets the version.
            request: The new version's create request.
        """
        app.versions.append(_FakeVersion(f"{app.id}-v{len(app.versions) + 1}", request))
        app.stale_gets_left = self.stale_gets

    def current(self, app: _FakeApp) -> AppVersionResponse:
        """Show ``app``'s current version as a list or a create shows it (not a GET).

        Args:
            app: The App whose latest version is shown.

        Returns:
            The latest version, its instance at the status the next GET reports.
        """
        version = app.versions[-1]
        return _version_response(version, self._status(version))

    def poll(self, app: _FakeApp) -> AppVersionResponse:
        """Report what one GET of ``app`` shows as its current version.

        Args:
            app: The App being fetched.

        Returns:
            The previous version as Running while stale GETs remain; otherwise
            the latest version, moved one status along.
        """
        if app.stale_gets_left:  # a redeploy Domino has not switched to yet
            app.stale_gets_left -= 1
            return _version_response(app.versions[-2], "Running")
        version = app.versions[-1]
        version.gets += 1
        return _version_response(version, self._status(version, offset=-1))

    def instances(
        self, app: _FakeApp, version_id: str
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
        status = self._status(version, offset=-1 if version.gets else 0)
        if version.id != version_id or status is None:
            return []
        return [_instance(version, status)]

    def version_of_instance(self, instance_id: str) -> _FakeVersion | None:
        """Find the version whose instance is ``instance_id``.

        Args:
            instance_id: An instance id, e.g. ``"app-1-v1-i"``.

        Returns:
            That version, or None if no App has it.
        """
        versions = (v for app in self.apps.values() for v in app.versions)
        return next((v for v in versions if _instance_id(v) == instance_id), None)

    def _status(self, version: _FakeVersion, offset: int = 0) -> str | None:
        return self.statuses[min(version.gets + offset, len(self.statuses) - 1)]


_VERSION_DEFAULTS: dict[str, object] = {
    "extendedIdentityPropagationToAppsEnabled": False,
    "environmentId": "",
    "environmentRevisionId": "",
    "hardwareTierId": "",
    "externalVolumeMountIds": [],
    "netAppVolumeIds": [],
    "tags": [],
}


def _instance_id(version: _FakeVersion) -> str:
    return f"{version.id}-i"


def _instance(version: _FakeVersion, status: str) -> AppInstanceSummaryResponse:
    return AppInstanceSummaryResponse(
        id=_instance_id(version), createdAt=0, status=status
    )


def _version_response(version: _FakeVersion, status: str | None) -> AppVersionResponse:
    instance = _instance(version, status) if status is not None else None
    fields = version.request.model_dump(exclude_none=True)
    return AppVersionResponse.model_validate(
        {**_VERSION_DEFAULTS, **fields, "id": version.id, "createdAt": 0}
        | {"updatedAt": 0, "currentInstance": instance}
    )
