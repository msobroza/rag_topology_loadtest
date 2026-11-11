"""What an App's instance reports: the one seam the fake's state calls out through.

``FakeDominoApps`` calls an ``InstanceScheduler`` when an App or version is
created and when an instance's status is read. The default, ``ScriptedInstances``,
scripts the statuses (the ``--behaviour`` scripts); another scheduler may be
supplied, e.g. one that runs each App somewhere real. Nothing here depends on
who supplies one.

Example:
    >>> FakeDominoApps(scheduler=ScriptedInstances(["Pending", "Running"]))
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import NamedTuple

from .api_shapes import AppCreationRequest, AppVersionCreationRequest


@dataclass
class FakeVersion:
    """One version of an App, as stored.

    Attributes:
        id: The version's id, ``app-<n>-v<m>``.
        request: What it was created with.
        gets: GETs of the App that reported this version (scripted statuses).
    """

    id: str
    request: AppVersionCreationRequest
    gets: int = 0


@dataclass
class FakeApp:
    """One App, as stored, its versions oldest first.

    Attributes:
        id: The App's id, ``app-<n>``.
        request: What it was created with.
        versions: Its versions; the last is the current one.
        stale_gets_left: GETs that still report the previous version (scripted).
    """

    id: str
    request: AppCreationRequest
    versions: list[FakeVersion]
    stale_gets_left: int = 0


def instance_id(version: FakeVersion) -> str:
    """The id of ``version``'s instance (one per version).

    Args:
        version: An App version.

    Returns:
        ``<version id>-i``, e.g. ``"app-1-v1-i"``.
    """
    return f"{version.id}-i"


class InstanceView(NamedTuple):
    """The version an App shows as current, and its instance's status."""

    version: FakeVersion
    status: str | None  # None: the version has no instance


class VersionRefusedError(ValueError):
    """A scheduler refuses an App version, which is then not stored.

    Args:
        status: The HTTP status the Apps API answers with.
        message: What is wrong, naming the offending value.
    """

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class InstanceScheduler(abc.ABC):
    """Decides what an App's instance reports.

    Example:
        >>> FakeDominoApps(scheduler=ScriptedInstances(["Running"]))
    """

    @abc.abstractmethod
    def version_added(self, app: FakeApp) -> None:
        """React to ``app`` getting a new current version (or being created).

        Args:
            app: The App; its last version is the new one.

        Raises:
            VersionRefusedError: If the version cannot be run; it is unstored.
        """

    @abc.abstractmethod
    def shown(self, app: FakeApp) -> InstanceView:
        """What a create or a list shows of ``app`` (not a GET: nothing moves).

        Args:
            app: The App.

        Returns:
            Its current version and its instance's status.
        """

    @abc.abstractmethod
    def polled(self, app: FakeApp) -> InstanceView:
        """What one GET of ``app`` shows; a GET may move the status along.

        Args:
            app: The App.

        Returns:
            Its current version and its instance's status.
        """

    @abc.abstractmethod
    def last_status(self, app: FakeApp) -> str | None:
        """The status of the latest version's instance, as the last GET left it.

        Args:
            app: The App.

        Returns:
            The status, or None if the instance does not exist.
        """


class ScriptedInstances(InstanceScheduler):
    """Today's scripted statuses: one per GET of the App, the last repeating.

    Args:
        statuses: What a new version's instance reports, one per GET of the App
            (None = no instance).
        stale_gets: GETs after a new version that still report the previous one
            as current and Running.

    Example:
        >>> ScriptedInstances(["Queued", "Running"], stale_gets=2)
    """

    def __init__(self, statuses: Sequence[str | None], stale_gets: int = 0) -> None:
        self.statuses, self.stale_gets = list(statuses), stale_gets

    def version_added(self, app: FakeApp) -> None:
        """Make a new version wait ``stale_gets`` GETs before it shows.

        Args:
            app: The App; its last version is the new one.
        """
        if len(app.versions) > 1:
            app.stale_gets_left = self.stale_gets

    def shown(self, app: FakeApp) -> InstanceView:
        """The latest version, its instance at the status the next GET reports.

        Args:
            app: The App.

        Returns:
            Its latest version and that status.
        """
        version = app.versions[-1]
        return InstanceView(version, self._status(version))

    def polled(self, app: FakeApp) -> InstanceView:
        """The previous version Running while stale GETs remain; else the next status.

        Args:
            app: The App being fetched.

        Returns:
            The version it shows and its instance's status.
        """
        if app.stale_gets_left:  # a redeploy Domino has not switched to yet
            app.stale_gets_left -= 1
            return InstanceView(app.versions[-2], "Running")
        version = app.versions[-1]
        version.gets += 1
        return InstanceView(version, self._status(version, offset=-1))

    def last_status(self, app: FakeApp) -> str | None:
        """The status the latest version's instance showed at the last GET.

        Args:
            app: The App.

        Returns:
            That status (the first one before any GET), or None for no instance.
        """
        version = app.versions[-1]
        return self._status(version, offset=-1 if version.gets else 0)

    def _status(self, version: FakeVersion, offset: int = 0) -> str | None:
        return self.statuses[min(version.gets + offset, len(self.statuses) - 1)]
