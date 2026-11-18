"""Tests of the internal route: every revision of an environment, or what it answered."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from fake_domino.server import create_fake_domino, fake_domino_http
from fake_domino.stand_ins import fake_environment
from fake_domino.state import FakeDominoApps, FakeEnvironment
from query_processing.deploy.domino_lookups.environments import EnvironmentRevision
from query_processing.deploy.domino_lookups.internal_revisions import (
    InternalRevisions,
)

BUILT_7 = "2026-11-12T09:41:00Z"
BUILT_6 = "2026-11-03T16:02:00Z"
SIX = [(n, f"rev-{n}", BUILT_6) for n in range(6, 0, -1)]


def http_on(apps: FakeDominoApps) -> httpx.Client:
    return fake_domino_http(create_fake_domino(apps))


def internal(
    *environments: FakeEnvironment,
) -> tuple[InternalRevisions, FakeDominoApps]:
    apps = FakeDominoApps(environments=environments)
    return InternalRevisions(http_on(apps)), apps


def test_every_revision_is_given_with_the_date_it_was_built() -> None:
    source, apps = internal(
        fake_environment("env-1", (7, "rev-7", BUILT_7), (6, "rev-6", BUILT_6))
    )
    assert source.revisions("env-1") == [
        EnvironmentRevision("rev-7", 7, datetime(2026, 11, 12, 9, 41, tzinfo=UTC)),
        EnvironmentRevision("rev-6", 6, datetime(2026, 11, 3, 16, 2, tzinfo=UTC)),
    ]
    assert source.describe("env-1") == (
        "the internal route GET /v4/environments/env-1/paginatedRevisionData lists "
        "2 revisions (7 to 6)"
    )
    assert apps.received == [("GET", "/v4/environments/env-1/paginatedRevisionData")]


ROUTE = "GET /v4/environments/env-1/paginatedRevisionData"


class AnswerOf:
    """Answers every call with HTTP 200 and one body, as a proxy or a Domino might."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __call__(self, _request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=self.body)


class RefusedConnection:
    """Refuses every call, as a host that does not listen would."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)


class TwoPagesWhateverIsAsked:
    """Answers two pages of five whatever page size is asked, counting the calls."""

    def __init__(self) -> None:
        self.asked: list[str | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.asked.append(request.url.params.get("pageSize"))
        page_info = {"totalPages": 2, "currentPage": 0, "pageSize": 5}
        return httpx.Response(200, json={"revisions": [], "pageInfo": page_info})


def through(handler: Callable[[httpx.Request], httpx.Response]) -> InternalRevisions:
    transport = httpx.MockTransport(handler)
    return InternalRevisions(
        httpx.Client(transport=transport, base_url="http://domino")
    )


def test_more_revisions_than_a_page_are_read_in_one_second_call() -> None:
    source, apps = internal(fake_environment("env-1", *SIX))
    assert [r.number for r in source.revisions("env-1")] == [6, 5, 4, 3, 2, 1]
    assert [r.number for r in source.revisions("env-1")] == [6, 5, 4, 3, 2, 1]
    assert (
        apps.received == [("GET", "/v4/environments/env-1/paginatedRevisionData")] * 2
    )


def test_a_route_that_answers_another_status_is_told() -> None:
    source, apps = internal(fake_environment("env-1", (7, "rev-7", BUILT_7)))
    apps.revisions_route_status = 404
    assert source.revisions("env-1") == []
    assert source.describe("env-1") == (
        f"the internal route {ROUTE} returned HTTP 404, so the revisions it would "
        "list are not known"
    )


def test_a_route_that_cannot_be_reached_is_told() -> None:
    source = through(RefusedConnection())
    assert source.revisions("env-1") == []
    assert source.describe("env-1").startswith(
        f"the internal route {ROUTE} could not be reached"
    )


@pytest.mark.parametrize(
    "body",
    [b"{}", b'{"revisions": "x", "pageInfo": {}}', b"<html>sign in</html>"],
    ids=["empty", "revisions-not-a-list", "not-json"],
)
def test_an_answer_of_another_shape_is_told(body: bytes) -> None:
    source = through(AnswerOf(body))
    assert source.revisions("env-1") == []
    assert source.describe("env-1") == (
        f"the internal route {ROUTE} answered in a shape other than the one Domino "
        "publishes (revisions with id, number and created; pageInfo with pageSize "
        "and totalPages)"
    )


def test_a_route_that_still_answers_several_pages_is_told() -> None:
    handler = TwoPagesWhateverIsAsked()
    source = through(handler)
    assert source.revisions("env-1") == []
    assert source.describe("env-1") == (
        f"the internal route {ROUTE} answered 2 pages when one page of 10 was "
        "asked, so its list is not complete"
    )
    assert handler.asked == [None, "10"]


def test_an_entry_without_its_id_its_number_or_its_date_is_left_out() -> None:
    entries = [
        {"id": "rev-3", "number": 3, "created": 1762940460000},
        {"number": 2, "created": 1762940460000},
        {"id": "rev-1", "number": "1", "created": 1762940460000},
        {"id": "rev-0", "number": 0},
    ]
    page_info = {"totalPages": 1, "currentPage": 0, "pageSize": 5}
    body = json.dumps({"revisions": entries, "pageInfo": page_info}).encode()
    source = through(AnswerOf(body))
    assert [revision.id for revision in source.revisions("env-1")] == ["rev-3"]
    assert source.describe("env-1") == (
        f"the internal route {ROUTE} lists 1 revision (3), leaving out 3 without an "
        "id, a number or a date"
    )
