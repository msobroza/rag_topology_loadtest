"""Tests of the environments: listed from the fake Domino with their three revisions."""

from __future__ import annotations

import pytest

from domino_deploy_helpers import BUILT_6, BUILT_7, http_on
from fake_domino.stand_ins import fake_environment
from fake_domino.state import FakeDominoApps
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.domino_lookups.environments import (
    ENVIRONMENTS_LIST_PATH,
    environment_from,
    list_environments,
)

# Not the helpers' FAKE_ENV_1: this one has a restricted revision.
ENV_1 = fake_environment(
    "env-1",
    (7, "rev-7", BUILT_7),
    (6, "rev-6", BUILT_6),
    (3, "rev-3", BUILT_6),
    name="qp-rag",
    active=6,
    restricted=3,
)


def test_the_environments_are_listed_with_their_three_revisions_and_every_field() -> (
    None
):
    [environment] = list_environments(http_on(FakeDominoApps(environments=[ENV_1])))
    assert (environment.id, environment.name) == ("env-1", "qp-rag")
    assert [r.number for r in environment.three()] == [7, 6, 3]
    assert environment.marks("rev-6") == ["active"]
    assert environment.fields["visibility"] == "private"


def test_one_revision_that_is_latest_and_active_is_given_once_with_both_marks() -> None:
    only = fake_environment("env-1", (7, "rev-7", BUILT_7))
    [environment] = list_environments(http_on(FakeDominoApps(environments=[only])))
    assert [r.id for r in environment.three()] == ["rev-7"]
    assert environment.marks("rev-7") == ["active", "latest"]


def test_an_environment_never_built_has_none_of_the_three() -> None:
    never = fake_environment("env-1")
    [environment] = list_environments(http_on(FakeDominoApps(environments=[never])))
    assert (environment.latest, environment.active, environment.restricted) == (
        None,
        None,
        None,
    )
    assert environment.three() == []


def test_eleven_environments_are_read_on_two_pages() -> None:
    apps = FakeDominoApps(
        environments=[fake_environment(f"env-{n}") for n in range(1, 12)]
    )
    assert len(list_environments(http_on(apps))) == 11
    assert apps.received == [("GET", ENVIRONMENTS_LIST_PATH)] * 2


@pytest.mark.parametrize("number", [None, "7", True])
def test_a_revision_without_a_whole_number_is_left_out(number: object) -> None:
    named = environment_from(
        {"id": "env-1", "latestRevision": {"id": "rev-7", "number": number}}
    )
    assert named.latest is None


def test_a_forbidden_list_says_the_sign_in_is_to_check() -> None:
    http = http_on(FakeDominoApps.from_behaviour("forbidden"))
    with pytest.raises(DominoDeployError) as refusal:
        list_environments(http)
    assert str(refusal.value) == (
        "Domino API GET /api/environments/beta/environments returned HTTP 403: "
        "Forbidden by the fake Domino (behaviour: forbidden); listing the "
        "environments needs no permission in Domino's reference: check the sign-in"
    )
