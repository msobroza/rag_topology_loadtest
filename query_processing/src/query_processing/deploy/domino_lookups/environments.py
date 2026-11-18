"""Domino's environments, each with the three revisions its documented API gives.

The list is ``GET /api/environments/beta/environments`` (no permission in Domino's
reference), whose answer holds ``environments`` and ``metadata.totalCount``. One
environment is ``GET /api/environments/v1/environments/{environmentId}``
(``ViewEnvironment``). An environment names at most three of its revisions:
``latestRevision``, ``selectedRevision`` (the active one) and
``restrictedRevision``, each with ``id``, ``number`` and ``status``.

Example:
    >>> [env.name for env in list_environments(http)]
    ['qp-rag']
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from ..apps_client import json_mapping
from .paged_list import read_pages

# Written here as well as in the fake Domino, which the product may not import.
ENVIRONMENTS_PATH = "/api/environments/v1/environments"
ENVIRONMENTS_LIST_PATH = "/api/environments/beta/environments"
_NEEDS = (
    "listing the environments needs no permission in Domino's reference: "
    "check the sign-in"
)


@dataclass(frozen=True)
class EnvironmentRevision:
    """One revision of an environment: its id, its number, when built (UTC), its status.

    The documented API tells no date: ``built`` is None for the revisions it gives.
    """

    id: str
    number: int
    built: datetime | None = None
    status: str = ""


@dataclass(frozen=True)
class DominoEnvironment:
    """An environment, its three revisions, and the environment as Domino gave it."""

    id: str
    name: str
    latest: EnvironmentRevision | None
    active: EnvironmentRevision | None
    restricted: EnvironmentRevision | None
    fields: Mapping[str, object] = field(default_factory=dict, compare=False)

    def three(self) -> list[EnvironmentRevision]:
        """Give the latest, active and restricted revisions there are, each id once.

        Returns:
            The present ones, in that order; one revision that is latest and active
            is given once.

        Example:
            >>> [revision.number for revision in environment.three()]
            [7, 6, 3]
        """
        present: list[EnvironmentRevision] = []
        for revision in (self.latest, self.active, self.restricted):
            if revision is not None and revision.id not in {r.id for r in present}:
                present.append(revision)
        return present

    def marks(self, revision_id: str) -> list[str]:
        """Say which of the three roles the revision ``revision_id`` holds.

        Args:
            revision_id: A revision's id.

        Returns:
            Of ``"active"``, ``"latest"`` and ``"restricted"``, in that order; empty
            for a revision that holds none.

        Example:
            >>> environment.marks("rev-7")
            ['active', 'latest']
        """
        roles = (
            ("active", self.active),
            ("latest", self.latest),
            ("restricted", self.restricted),
        )
        return [
            mark for mark, held in roles if held is not None and held.id == revision_id
        ]


def environment_from(fields: Mapping[str, object]) -> DominoEnvironment:
    """Read an ``EnvironmentV1`` of Domino's answer, leniently, as every answer is.

    Args:
        fields: The environment as Domino gave it.

    Returns:
        The environment; a revision Domino does not name, or names without a
        whole ``number``, is None.

    Example:
        >>> environment_from({"id": "env-1", "name": "qp-rag"}).latest is None
        True
    """
    return DominoEnvironment(
        id=str(fields.get("id", "")),
        name=str(fields.get("name", "")),
        latest=_revision_from(fields.get("latestRevision")),
        active=_revision_from(fields.get("selectedRevision")),
        restricted=_revision_from(fields.get("restrictedRevision")),
        fields=dict(fields),
    )


def list_environments(http: httpx.Client) -> list[DominoEnvironment]:
    """Read every environment the person can see, in Domino's order.

    Args:
        http: Client whose base URL is Domino (``DOMINO_API_PROXY`` in a run).

    Returns:
        The environments, each with its three revisions.

    Raises:
        DominoDeployError: If the list cannot be read; on 401 or 403 the message
            says the list needs no permission, so the sign-in is to check.

    Example:
        >>> [environment.id for environment in list_environments(http)]
        ['env-1']
    """
    listed = read_pages(http, ENVIRONMENTS_LIST_PATH, "environments", needs=_NEEDS)
    return [environment_from(environment) for environment in listed]


def _revision_from(value: object) -> EnvironmentRevision | None:
    revision = json_mapping(value)
    number = revision.get("number")
    # A revision without a whole number is left out: read as 0, it would answer
    # an environment_revision of 0 that no revision of Domino has.
    if isinstance(number, bool) or not isinstance(number, int):
        return None
    return EnvironmentRevision(
        id=str(revision.get("id", "")),
        number=number,
        status=str(revision.get("status") or ""),
    )
