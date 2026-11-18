"""Tests of the environments shown: their table, an environment's revisions, the details.

On the fake Domino, whose environment env-1 has the revisions 7 (latest), 6
(active) and 5, the last two built the same day.
"""

from __future__ import annotations

import httpx
import yaml

from fake_domino.server import create_fake_domino, fake_domino_http
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

BUILT_7 = "2026-11-12T09:41:00Z"
BUILT_6 = "2026-11-03T16:02:00Z"
ENV_1 = fake_environment(
    "env-1",
    (7, "rev-7", BUILT_7),
    (6, "rev-6", BUILT_6),
    (5, "rev-5", BUILT_6),
    name="qp-rag",
    active=6,
)
ROUTE_404 = (
    "the internal route GET /v4/environments/env-1/paginatedRevisionData returned "
    "HTTP 404, so the revisions it would list are not known"
)


def http_on(apps: FakeDominoApps) -> httpx.Client:
    return fake_domino_http(create_fake_domino(apps))


def env_1_on(apps: FakeDominoApps) -> DominoEnvironment:
    [environment] = list_environments(http_on(apps))
    return environment


def test_the_environments_table_gives_the_active_and_latest_numbers() -> None:
    apps = FakeDominoApps(environments=[ENV_1, fake_environment("env-2", name="empty")])
    assert environments_table(list_environments(http_on(apps))) == (
        "NAME    VISIBILITY  OWNER       ACTIVE  LATEST  ID\n"
        "qp-rag  private     fake-owner       6       7  env-1\n"
        "empty   private     fake-owner       -       -  env-2\n"
    )


def test_the_revisions_table_marks_each_revision_and_dates_it_in_utc() -> None:
    [environment] = list_environments(http_on(FakeDominoApps(environments=[ENV_1])))
    revisions, note = shown_revisions(
        environment, InternalRevisions(http_on(FakeDominoApps(environments=[ENV_1])))
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
    apps = FakeDominoApps(environments=[ENV_1])
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
    apps = FakeDominoApps(environments=[ENV_1])
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
        environments_details([env_1_on(FakeDominoApps(environments=[ENV_1]))], {})
    )
    assert "revisions" not in shown
    assert shown["latestRevision"]["number"] == 7


def test_the_details_of_a_failed_route_carry_its_note() -> None:
    apps = FakeDominoApps(environments=[ENV_1])
    apps.revisions_route_status = 404
    environment = env_1_on(apps)
    given = shown_revisions(environment, InternalRevisions(http_on(apps)))
    [shown] = yaml.safe_load(environments_details([environment], {"env-1": given}))
    assert [revision["built"] for revision in shown["revisions"]] == [None, None]
    assert shown["revisions_note"] == ROUTE_404
