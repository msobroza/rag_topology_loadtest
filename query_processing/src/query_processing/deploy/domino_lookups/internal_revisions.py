"""Every revision of an environment, through the internal route Domino's own pages use.

The route is not in Domino's reference: its path is remembered, not published.
Its shape is the ``PaginatedRevisionData`` of the published 6.3.1 spec:
``revisions``, each a ``RevisionSummary`` with ``id``, ``number``, ``created``
(an epoch) and ``active``, and ``pageInfo`` with ``totalPages``,
``currentPage`` and ``pageSize``. The path, the parameter ``pageSize`` and the
unit of ``created`` (taken as milliseconds since 1970, UTC) are not verified on
a live Domino, nor that the route answers through ``DOMINO_API_PROXY``; the
design's "The one check on a live Domino" tells how to check them.

This source never raises: when the route fails it gives no revision, and
``describe`` says what the route answered.

Example:
    >>> [revision.number for revision in InternalRevisions(http).revisions("env-1")]
    [7, 6, 5]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from urllib.parse import quote

import httpx

from ...domain.errors import describe_error
from ..apps_client import (
    DOMINO_API,
    DominoDeployError,
    json_list,
    json_mapping,
    send_to_domino,
)
from .environments import EnvironmentRevision
from .revision_sources import BaseRevisionSource

# Written here as well as in the fake Domino, which the product may not import.
REVISIONS_ROUTE = "/v4/environments/{environment_id}/paginatedRevisionData"
_SHAPE = (
    "answered in a shape other than the one Domino publishes (revisions with id, "
    "number and created; pageInfo with pageSize and totalPages)"
)


class _RouteFailedError(Exception):
    """The route failed; the message is the clause that tells what it answered."""


@dataclass(frozen=True)
class _RouteReading:
    """What one environment's route gave: its revisions and the clause telling it."""

    revisions: list[EnvironmentRevision]
    told: str
    left_out: int = 0


@dataclass(frozen=True)
class _RoutePage:
    """One answer of the route: its entries, its page size and its number of pages."""

    entries: list[object]
    page_size: int
    total_pages: int


class InternalRevisions(BaseRevisionSource):
    """The internal route for one environment: every revision, asked once.

    Nothing is asked before a revision or a description is.

    Args:
        http: Client whose base URL is Domino (``DOMINO_API_PROXY`` in a run).

    Example:
        >>> source = InternalRevisions(http)
    """

    def __init__(self, http: httpx.Client) -> None:
        self._http = http
        self._read: dict[str, _RouteReading] = {}

    def revisions(self, environment_id: str) -> list[EnvironmentRevision]:
        """Give every revision of the environment, newest first, with its build date.

        Args:
            environment_id: The environment's id.

        Returns:
            The revisions, ``built`` read from ``created`` (None when it cannot
            be read); an empty list when the route fails (``describe`` says how).
            An entry without its id or a whole number is left out, and
            ``describe`` counts it.

        Example:
            >>> [revision.number for revision in source.revisions("env-1")]
            [7, 6, 5]
        """
        return list(self._reading(environment_id).revisions)

    def left_out(self, environment_id: str) -> int:
        """Count the entries the route gave without an id or a whole number.

        Args:
            environment_id: The environment's id.

        Returns:
            How many entries ``revisions`` left out; ``describe`` says it too.

        Example:
            >>> source.left_out("env-1")
            0
        """
        return self._reading(environment_id).left_out

    def describe(self, environment_id: str) -> str:
        """Say how many revisions the route listed, or what it answered instead.

        Args:
            environment_id: The environment's id.

        Returns:
            E.g. ``"the internal route GET /v4/environments/env-1/paginatedRevisionData
            lists 2 revisions (7 to 6)"``.

        Example:
            >>> source.describe("env-1")[-24:]
            'lists 2 revisions (7 to 6)'
        """
        return self._reading(environment_id).told

    def _reading(self, environment_id: str) -> _RouteReading:
        if environment_id not in self._read:
            self._read[environment_id] = _read_route(self._http, environment_id)
        return self._read[environment_id]


def _read_route(http: httpx.Client, environment_id: str) -> _RouteReading:
    path = REVISIONS_ROUTE.format(environment_id=quote(environment_id, safe=""))
    route = f"the internal route GET {path}"
    try:
        entries = _every_entry(http, path)
    except _RouteFailedError as failure:
        return _RouteReading([], f"{route} {failure}")
    found = (_entry_revision(entry) for entry in entries)
    revisions = [revision for revision in found if revision is not None]
    revisions.sort(key=lambda revision: revision.number, reverse=True)
    left_out = len(entries) - len(revisions)
    told = _listed(revisions, left_out=left_out)
    return _RouteReading(revisions, f"{route} {told}", left_out)


def _every_entry(http: httpx.Client, path: str) -> list[object]:
    # No assumption on how pages are numbered: the first answer tells the page
    # size and the number of pages, and a second call asks one page of them all.
    page = _read_route_page(http, path, params=None)
    if page.total_pages > 1:
        asked = page.page_size * page.total_pages
        page = _read_route_page(http, path, params={"pageSize": str(asked)})
        if page.total_pages > 1:
            raise _RouteFailedError(
                f"answered {page.total_pages} pages when one page of {asked} was "
                "asked, so its list is not complete"
            )
    return page.entries


def _read_route_page(
    http: httpx.Client, path: str, *, params: Mapping[str, str] | None
) -> _RoutePage:
    try:
        response = send_to_domino(http, "GET", path, api=DOMINO_API, params=params)
    except DominoDeployError as exc:
        # Domino's message would repeat the route: only the cause is told.
        cause = exc.__cause__ or exc
        base_url = str(http.base_url).rstrip("/")
        raise _RouteFailedError(
            f"could not be reached at {base_url}: {describe_error(cause)}"
        ) from exc
    if response.status_code != HTTPStatus.OK:
        raise _RouteFailedError(
            f"returned HTTP {response.status_code}, so the revisions it would list "
            "are not known"
        )
    try:
        body: object = response.json()
    except ValueError as exc:
        raise _RouteFailedError(_SHAPE) from exc
    return _page_from(json_mapping(body))


def _page_from(body: Mapping[str, object]) -> _RoutePage:
    entries = body.get("revisions")
    info = json_mapping(body.get("pageInfo"))
    page_size = _whole(info.get("pageSize"), least=1)
    total_pages = _whole(info.get("totalPages"), least=0)
    if not isinstance(entries, list) or page_size is None or total_pages is None:
        raise _RouteFailedError(_SHAPE)
    return _RoutePage(json_list(entries), page_size, total_pages)


def _whole(value: object, *, least: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        return None
    return value


def _entry_revision(entry: object) -> EnvironmentRevision | None:
    # An entry lacking its id or a whole number is left out, as environments.py
    # leaves out a revision whose number is no whole number; describe counts
    # them, so a shape Domino changed inside the entries shows. A date that
    # cannot be read is no reason: a deploy pins by id and number alone.
    fields = json_mapping(entry)
    revision_id = fields.get("id")
    number = _whole(fields.get("number"), least=1)
    if not revision_id or number is None:
        return None
    built = _built_at(fields.get("created"))
    return EnvironmentRevision(id=str(revision_id), number=number, built=built)


def _built_at(created: object) -> datetime | None:
    if isinstance(created, bool) or not isinstance(created, int | float):
        return None
    try:
        return datetime.fromtimestamp(created / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):  # an epoch no date can hold
        return None


def _listed(revisions: list[EnvironmentRevision], *, left_out: int) -> str:
    skipped = ""
    if left_out:
        skipped = f", leaving out {left_out} without an id or a whole number"
    if not revisions:
        return f"lists no revision{skipped}"
    if len(revisions) == 1:
        return f"lists 1 revision ({revisions[0].number}){skipped}"
    newest, oldest = revisions[0].number, revisions[-1].number
    return f"lists {len(revisions)} revisions ({newest} to {oldest}){skipped}"
