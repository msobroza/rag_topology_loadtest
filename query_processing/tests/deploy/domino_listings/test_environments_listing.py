"""Tests of the environments shown: their table, an environment's revisions, the details.

On the fake Domino, whose environment env-1 has the revisions 7 (latest), 6
(active) and 5, the last two built the same day.
"""

from __future__ import annotations

import json

import httpx
import yaml

from domino_deploy_helpers import FAKE_ENV_1, http_on
from fake_domino.stand_ins import fake_environment
from fake_domino.state import FakeDominoApps
from query_processing.deploy.domino_listings.environments_listing import (
    environments_details,
    environments_table,
    revisions_table,
    shown_revisions,
)
from query_processing.deploy.domino_lookups.environments import (
    DominoEnvironment,
    list_environments,
)
from query_processing.deploy.domino_lookups.internal_revisions import (
    InternalRevisions,
)

ROUTE_404 = (
    "the internal route GET /v4/environments/env-1/paginatedRevisionData returned "
    "HTTP 404, so the revisions it would list are not known"
)


def env_1_on(apps: FakeDominoApps) -> DominoEnvironment:
    [environment] = list_environments(http_on(apps))
    return environment


def test_the_environments_table_gives_the_active_and_latest_numbers() -> None:
    apps = FakeDominoApps(
        environments=[FAKE_ENV_1, fake_environment("env-2", name="empty")]
    )
    assert environments_table(list_environments(http_on(apps))) == (
        "NAME    VISIBILITY  OWNER       ACTIVE  LATEST  ID\n"
        "qp-rag  private     fake-owner       6       7  env-1\n"
        "empty   private     fake-owner       -       -  env-2\n"
    )


def test_the_revisions_table_marks_each_revision_and_dates_it_in_utc() -> None:
    [environment] = list_environments(
        http_on(FakeDominoApps(environments=[FAKE_ENV_1]))
    )
    revisions, note = shown_revisions(
        environment,
        InternalRevisions(http_on(FakeDominoApps(environments=[FAKE_ENV_1]))),
    )
    assert note == ""
    assert revisions_table(environment, revisions, note) == (
        "qp-rag   id env-1   private, owner fake-owner\n"
        "\n"
        "REVISION  BUILT (UTC)       STATUS     ID\n"
        "7 latest  2026-11-12 09:41  succeeded  rev-7\n"
        "6 active  2026-11-03 16:02  succeeded  rev-6\n"
        "5         2026-11-03 16:02             rev-5\n"
    )


def test_a_route_that_fails_leaves_the_three_documented_revisions_and_a_note() -> None:
    apps = FakeDominoApps(environments=[FAKE_ENV_1])
    apps.revisions_route_status = 404
    environment = env_1_on(apps)
    revisions, note = shown_revisions(environment, InternalRevisions(http_on(apps)))
    assert note == ROUTE_404
    assert revisions_table(environment, revisions, note) == (
        "qp-rag   id env-1   private, owner fake-owner\n"
        "\n"
        "REVISION  BUILT (UTC)  STATUS     ID\n"
        "7 latest  -            succeeded  rev-7\n"
        "6 active  -            succeeded  rev-6\n"
        "\n"
        f"{ROUTE_404}\n"
    )


def test_the_environment_details_hold_its_revisions_when_given() -> None:
    apps = FakeDominoApps(environments=[FAKE_ENV_1])
    environment = env_1_on(apps)
    revisions, _ = shown_revisions(environment, InternalRevisions(http_on(apps)))
    [shown] = yaml.safe_load(
        environments_details([environment], {"env-1": (revisions, "")})
    )
    assert next(iter(shown)) == "name"
    assert shown["revisions"][0] == {
        "number": 7,
        "id": "rev-7",
        "built": "2026-11-12 09:41 UTC",
        "marks": ["latest"],
        "status": "succeeded",
    }


def test_the_environment_details_hold_no_revisions_unless_given() -> None:
    [shown] = yaml.safe_load(
        environments_details([env_1_on(FakeDominoApps(environments=[FAKE_ENV_1]))], {})
    )
    assert "revisions" not in shown
    assert shown["latestRevision"]["number"] == 7


def test_the_details_of_a_failed_route_carry_its_note() -> None:
    apps = FakeDominoApps(environments=[FAKE_ENV_1])
    apps.revisions_route_status = 404
    environment = env_1_on(apps)
    given = shown_revisions(environment, InternalRevisions(http_on(apps)))
    [shown] = yaml.safe_load(environments_details([environment], {"env-1": given}))
    assert [revision["built"] for revision in shown["revisions"]] == [None, None]
    assert shown["revisions_note"] == ROUTE_404


class RouteLeavingOutOneEntry:
    """Answers the internal route with rev-7 and an entry without its id."""

    def __call__(self, _request: httpx.Request) -> httpx.Response:
        entries = [
            {"id": "rev-7", "number": 7, "created": 1762940460000},
            {"number": 6},
        ]
        page_info = {"totalPages": 1, "currentPage": 0, "pageSize": 5}
        body = {"revisions": entries, "pageInfo": page_info}
        return httpx.Response(200, content=json.dumps(body).encode())


def test_a_route_that_left_out_entries_is_shown_with_its_note() -> None:
    environment = env_1_on(FakeDominoApps(environments=[FAKE_ENV_1]))
    route = httpx.MockTransport(RouteLeavingOutOneEntry())
    internal = InternalRevisions(httpx.Client(transport=route, base_url="http://d"))
    revisions, note = shown_revisions(environment, internal)
    assert [revision.id for revision in revisions] == ["rev-7"]
    assert note == (
        "the internal route GET /v4/environments/env-1/paginatedRevisionData lists "
        "1 revision (7), leaving out 1 without an id or a whole number"
    )
    assert revisions_table(environment, revisions, note).endswith(f"\n\n{note}\n")
    given = {"env-1": (revisions, note)}
    [shown] = yaml.safe_load(environments_details([environment], given))
    assert shown["revisions_note"] == note
