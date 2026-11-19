"""Tests of finding a revision by its number: the documented call, then later sources."""

from __future__ import annotations

import httpx
import pytest

from domino_deploy_helpers import BUILT_6, BUILT_7, WHERE, http_on
from fake_domino.stand_ins import fake_environment
from fake_domino.state import FakeDominoApps, FakeEnvironment
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.domino_lookups.environments import (
    ENVIRONMENTS_PATH,
    EnvironmentRevision,
)
from query_processing.deploy.domino_lookups.revision_sources import (
    BaseRevisionSource,
    DocumentedRevisions,
    revision_numbered,
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


class RevisionsOf(BaseRevisionSource):
    """A later source holding the revisions it is given, counting its calls."""

    def __init__(self, *revisions: EnvironmentRevision) -> None:
        self.given, self.calls = list(revisions), 0

    def revisions(self, environment_id: str) -> list[EnvironmentRevision]:
        self.calls += 1
        return self.given

    def describe(self, environment_id: str) -> str:
        return f"the named fake gives {len(self.given)} revisions"


def documented(
    *environments: FakeEnvironment,
) -> tuple[DocumentedRevisions, FakeDominoApps]:
    apps = FakeDominoApps(environments=environments)
    return DocumentedRevisions(http_on(apps)), apps


def refusal_of(source: BaseRevisionSource, environment_id: str, number: int) -> str:
    with pytest.raises(DominoDeployError) as refusal:
        revision_numbered([source], environment_id, number, WHERE)
    return str(refusal.value)


@pytest.mark.parametrize(
    ("number", "revision_id"), [(7, "rev-7"), (6, "rev-6"), (3, "rev-3")]
)
def test_the_latest_the_active_and_the_restricted_revisions_are_found(
    number: int, revision_id: str
) -> None:
    source, _ = documented(ENV_1)
    found = revision_numbered([source], "env-1", number, WHERE)
    assert (found.id, found.number, found.status) == (revision_id, number, "succeeded")


def test_an_environment_is_asked_once_however_many_numbers_are_looked_up() -> None:
    source, apps = documented(ENV_1)
    for number in (7, 6):
        revision_numbered([source], "env-1", number, WHERE)
    assert apps.received == [("GET", f"{ENVIRONMENTS_PATH}/env-1")]


def test_a_later_source_is_asked_only_for_a_number_the_first_lacks() -> None:
    source, _ = documented(ENV_1)
    older = RevisionsOf(EnvironmentRevision("rev-5", 5))
    assert revision_numbered([source, older], "env-1", 7, WHERE).id == "rev-7"
    assert older.calls == 0
    assert revision_numbered([source, older], "env-1", 5, WHERE).id == "rev-5"


def test_a_number_no_source_has_is_refused_with_what_each_source_holds() -> None:
    source, _ = documented(ENV_1)
    with pytest.raises(DominoDeployError) as refusal:
        revision_numbered([source, RevisionsOf()], "env-1", 5, WHERE)
    assert str(refusal.value) == (
        f"environment_revision 5 of {WHERE} is no revision Domino gives for "
        "environment 'env-1': the documented API gives its latest (7), active (6) "
        "and restricted (3) revisions; the named fake gives 0 revisions. Write the "
        "revision's id as environment_revision_id"
    )


def test_an_environment_never_built_says_none_three_times() -> None:
    source, _ = documented(fake_environment("env-1"))
    assert "latest (none), active (none) and restricted (none)" in refusal_of(
        source, "env-1", 1
    )


def test_an_unknown_environment_is_refused_with_its_id() -> None:
    source, _ = documented(ENV_1)
    assert refusal_of(source, "env-9", 1) == (
        "Domino has no environment 'env-9' (GET /api/environments/v1/environments/"
        "env-9 returned HTTP 404): check environment_id in deploy.yaml"
    )


def test_a_forbidden_environment_names_the_permission() -> None:
    source = DocumentedRevisions(http_on(FakeDominoApps.from_behaviour("forbidden")))
    assert refusal_of(source, "env-1", 7) == (
        "Domino API GET /api/environments/v1/environments/env-1 returned HTTP 403: "
        "Forbidden by the fake Domino (behaviour: forbidden); reading an environment "
        "needs the permission ViewEnvironment: ask for it, or write the revision's "
        "id as environment_revision_id, which needs no such call"
    )


def test_the_documented_source_gives_its_environment() -> None:
    source, _ = documented(ENV_1)
    assert source.environment("env-1").name == "qp-rag"


class EnvelopeWithoutEnvironment:
    """Answers 200 with JSON of another shape, as a proxy's own page might."""

    def __call__(self, _request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"metadata": {}})


def test_an_answer_without_an_environment_is_refused_and_ends_the_lookup() -> None:
    http = httpx.Client(
        transport=httpx.MockTransport(EnvelopeWithoutEnvironment()),
        base_url="http://domino",
    )
    later = RevisionsOf(EnvironmentRevision("rev-7", 7))
    with pytest.raises(DominoDeployError) as refusal:
        revision_numbered([DocumentedRevisions(http), later], "env-1", 7, WHERE)
    assert str(refusal.value) == (
        "Domino API GET /api/environments/v1/environments/env-1 answered HTTP 200 "
        "without an environment (an EnvironmentEnvelopeV1 was expected): "
        '{"metadata":{}}'
    )
    assert later.calls == 0
