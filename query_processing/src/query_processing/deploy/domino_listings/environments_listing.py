"""Domino's environments shown: their table, one's revisions, or every field as YAML.

An environment's revisions come from the internal route, each with the status
the documented API gives its latest, active and restricted revisions; when the
route fails, those three are shown with a note telling what it answered.

Example:
    >>> print(environments_table(list_environments(http)), end="")
    NAME    VISIBILITY  OWNER       ACTIVE  LATEST  ID
    qp-rag  private     fake-owner       6       7  env-1
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from datetime import datetime

from ..apps_client import json_mapping
from ..domino_lookups.environments import DominoEnvironment, EnvironmentRevision
from ..domino_lookups.revision_sources import BaseRevisionSource
from .aligned_table import aligned_table
from .listing_names import details_yaml

_HEADERS = ("NAME", "VISIBILITY", "OWNER", "ACTIVE", "LATEST", "ID")
_RIGHT = (3, 4)  # ACTIVE and LATEST, numbers
_REVISION_HEADERS = ("REVISION", "BUILT (UTC)", "STATUS", "ID")
_BUILT = "%Y-%m-%d %H:%M"


def environments_table(environments: Sequence[DominoEnvironment]) -> str:
    """Lay out the environments in Domino's order, one line each.

    Args:
        environments: The environments (see ``list_environments``).

    Returns:
        The table, with the numbers of each one's active and latest revisions,
        ``-`` for a revision it lacks.

    Example:
        >>> environments_table(environments).splitlines()[1]
        'qp-rag  private     fake-owner       6       7  env-1'
    """
    rows = [
        [
            environment.name,
            _visibility(environment),
            _owner(environment),
            _number(environment.active),
            _number(environment.latest),
            environment.id,
        ]
        for environment in environments
    ]
    return aligned_table(_HEADERS, rows, right=_RIGHT)


def shown_revisions(
    environment: DominoEnvironment, internal: BaseRevisionSource
) -> tuple[list[EnvironmentRevision], str]:
    """Give the revisions to show, and a note when the internal source lacks some.

    Args:
        environment: The environment, with its documented three revisions.
        internal: The source of every revision (``InternalRevisions``).

    Returns:
        Every revision of ``internal``, each with the status the documented three
        give it, and ``""``, or what ``internal`` describes when it left entries
        out; or, when ``internal`` has none, the documented three and what
        ``internal`` describes.

    Example:
        >>> revisions, note = shown_revisions(environment, InternalRevisions(http))
        >>> [revision.number for revision in revisions], note
        ([7, 6, 5], '')
    """
    every = internal.revisions(environment.id)
    if not every:
        return environment.three(), internal.describe(environment.id)
    statuses = {revision.id: revision.status for revision in environment.three()}
    left_out = internal.left_out(environment.id)
    return [
        dataclasses.replace(revision, status=statuses.get(revision.id, ""))
        for revision in every
    ], (internal.describe(environment.id) if left_out else "")


def revisions_table(
    environment: DominoEnvironment, revisions: Sequence[EnvironmentRevision], note: str
) -> str:
    """Lay out an environment's line, then its revisions with their marks and dates.

    Args:
        environment: The environment.
        revisions: Its revisions to show (see ``shown_revisions``), newest first.
        note: What the internal route answered in place of its revisions, or
            ``""``.

    Returns:
        The environment's id, visibility and owner, a blank line, the table of
        revisions (dates in UTC, ``-`` when not known), and the note after a
        blank line when there is one.

    Example:
        >>> print(revisions_table(environment, revisions, "").splitlines()[3])
        7 latest  2026-11-12 09:41  succeeded  rev-7
    """
    heading = (
        f"{environment.name}   id {environment.id}   "
        f"{_visibility(environment)}, owner {_owner(environment)}\n\n"
    )
    rows = [
        [
            " ".join([str(revision.number), *environment.marks(revision.id)]),
            _built(revision),
            revision.status,
            revision.id,
        ]
        for revision in revisions
    ]
    table = heading + aligned_table(_REVISION_HEADERS, rows)
    return f"{table}\n{note}\n" if note else table


def revision_lines(
    environment: DominoEnvironment, revisions: Sequence[EnvironmentRevision]
) -> list[str]:
    """Write one line a revision, as an open environment of the list shows them.

    Args:
        environment: The environment, whose three revisions give the marks.
        revisions: Its revisions to show (see ``shown_revisions``), newest first.

    Returns:
        The lines, columns aligned: the number, the marks, the build date in UTC
        (``-`` when not known) and the id.

    Example:
        >>> revision_lines(environment, revisions)[0]
        'v7  latest  built 2026-11-12 09:41  id rev-7'
    """
    rows = [
        [
            f"v{revision.number}",
            " ".join(environment.marks(revision.id)),
            f"built {_built(revision)}",
            f"id {revision.id}",
        ]
        for revision in revisions
    ]
    return aligned_table(("", "", "", ""), rows).splitlines()[1:]


def environments_details(
    environments: Sequence[DominoEnvironment],
    revisions: Mapping[str, tuple[list[EnvironmentRevision], str]],
) -> str:
    """Write every field Domino gave for each environment, as YAML, the name first.

    Args:
        environments: The environments (see ``list_environments``).
        revisions: By environment id, the revisions and note of
            ``shown_revisions``; an environment given here gains ``revisions``
            (and ``revisions_note`` when the note is not empty).

    Returns:
        A YAML list, in Domino's order.

    Example:
        >>> print(environments_details(environments, {}).splitlines()[0])
        - name: qp-rag
    """
    items = [
        _environment_fields(environment, revisions.get(environment.id))
        for environment in environments
    ]
    return details_yaml(items)


def _environment_fields(
    environment: DominoEnvironment,
    revisions: tuple[list[EnvironmentRevision], str] | None,
) -> dict[str, object]:
    fields = dict(environment.fields)
    if revisions is None:
        return fields
    shown, note = revisions
    fields["revisions"] = [_revision_fields(environment, r) for r in shown]
    if note:
        fields["revisions_note"] = note
    return fields


def _revision_fields(
    environment: DominoEnvironment, revision: EnvironmentRevision
) -> dict[str, object]:
    return {
        "number": revision.number,
        "id": revision.id,
        "built": _built_utc(revision.built),
        "marks": environment.marks(revision.id),
        "status": revision.status,
    }


def _built(revision: EnvironmentRevision) -> str:
    return "-" if revision.built is None else revision.built.strftime(_BUILT)


def _built_utc(built: datetime | None) -> str | None:
    return None if built is None else f"{built.strftime(_BUILT)} UTC"


def _number(revision: EnvironmentRevision | None) -> str:
    return "-" if revision is None else str(revision.number)


def _visibility(environment: DominoEnvironment) -> str:
    return str(environment.fields.get("visibility", ""))


def _owner(environment: DominoEnvironment) -> str:
    return str(json_mapping(environment.fields.get("owner")).get("username", ""))
