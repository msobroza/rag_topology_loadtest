"""The ids behind a tier's name and a revision's number, found before any App is published.

Through ``deploy`` and ``main`` over the fake Domino, which has two tiers and one
environment with three revisions: what is sent and printed, which calls are
made, and each refusal, before anything is created. The fixtures (run_main) are
in domino_deploy_fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from domino_deploy_helpers import (
    ENV_1,
    LAUNCH,
    RunMain,
    make_client,
    make_settings,
    no_sleep,
)
from fake_domino.server import create_fake_domino
from fake_domino.stand_ins import fake_environment, fake_tier
from fake_domino.state import FakeDominoApps
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.publish import deploy

pytest_plugins = ["domino_deploy_fixtures"]

BUILT_7 = "2026-11-12T09:41:00Z"
BUILT_6 = "2026-11-03T16:02:00Z"
TIERS = [
    fake_tier("small-k8s", "cpu-8", 8, 32),
    fake_tier("large-k8s", "cpu-16", 16, 64),
]
# Not ENV_1: the helpers' ENV_1 is the text of a deploy file, which a test writes.
FAKE_ENV_1 = fake_environment(
    "env-1", (7, "rev-7", BUILT_7), (6, "rev-6", BUILT_6), (5, "rev-5", BUILT_6)
)
TIERS_CALL = ("GET", "/api/hardwaretiers/v1/hardwaretiers")
ENVIRONMENT_CALL = ("GET", "/api/environments/v1/environments/env-1")
INTERNAL_CALL = ("GET", "/v4/environments/env-1/paginatedRevisionData")
LOOKUPS = {TIERS_CALL, ENVIRONMENT_CALL, INTERNAL_CALL}
NAMED = {"environment_id": "env-1", "environment_revision": 7, "hardware_tier": "cpu-8"}


def fake_with_tiers_and_environment() -> FakeDominoApps:
    return FakeDominoApps(tiers=TIERS, environments=[FAKE_ENV_1])


def deployed(target: str = "all", **given: object) -> FakeDominoApps:
    apps = fake_with_tiers_and_environment()
    settings = make_settings(**given)
    deploy(make_client(apps), settings, target, launch=LAUNCH, sleep=no_sleep)
    return apps


def lookups_of(apps: FakeDominoApps) -> list[tuple[str, str]]:
    return [call for call in apps.received if call in LOOKUPS]


def test_a_tier_and_a_revision_by_name_are_sent_as_their_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    apps = deployed(defaults=NAMED, apps={"reranker": {"hardware_tier": "cpu-16"}})
    sent = {
        c.name: (c.version.hardwareTierId, c.version.environmentRevisionId)
        for c in apps.creates
    }
    assert sent == {
        "rag-embedder": ("small-k8s", "rev-7"),
        "rag-reranker": ("large-k8s", "rev-7"),
        "rag-workflow": ("small-k8s", "rev-7"),
    }
    assert capsys.readouterr().out.startswith(
        "rag-embedder: hardware tier cpu-8 (id small-k8s): 8 cores, 32 GiB\n"
        "rag-embedder: environment revision 7 (id rev-7)\n"
        "rag-reranker: hardware tier cpu-16 (id large-k8s): 16 cores, 64 GiB\n"
        "rag-reranker: environment revision 7 (id rev-7)\n"
        "rag-workflow: hardware tier cpu-8 (id small-k8s): 8 cores, 32 GiB\n"
        "rag-workflow: environment revision 7 (id rev-7)\n"
    )
    assert lookups_of(apps) == [TIERS_CALL, ENVIRONMENT_CALL]


def test_an_older_revision_is_found_through_the_internal_route() -> None:
    apps = deployed(defaults={"environment_id": "env-1", "environment_revision": 5})
    assert {c.version.environmentRevisionId for c in apps.creates} == {"rev-5"}
    assert lookups_of(apps) == [ENVIRONMENT_CALL, INTERNAL_CALL]


@pytest.mark.parametrize(
    ("by_name", "by_id"),
    [
        ({"hardware_tier": "cpu-8"}, {"hardware_tier_id": "small-k8s"}),
        ({"environment_revision": 7}, {"environment_revision_id": "rev-7"}),
    ],
)
def test_a_setting_written_either_way_publishes_the_same_versions(
    by_name: dict[str, object], by_id: dict[str, object]
) -> None:
    named = deployed(defaults={"environment_id": "env-1", **by_name})
    with_ids = deployed(defaults={"environment_id": "env-1", **by_id})
    assert named.creates == with_ids.creates


@pytest.mark.parametrize(
    ("wrong", "message"),
    [
        ({"hardware_tier": "cpu-61"}, "hardware_tier 'cpu-61' of App rag-workflow"),
        ({"environment_revision": 9}, "environment_revision 9 of App rag-workflow"),
    ],
)
def test_a_wrong_value_of_the_last_app_publishes_nothing(
    wrong: dict[str, object], message: str
) -> None:
    apps = fake_with_tiers_and_environment()
    settings = make_settings(defaults=NAMED, apps={"workflow": wrong})
    with pytest.raises(DominoDeployError, match=message):
        deploy(make_client(apps), settings, "all", launch=LAUNCH, sleep=no_sleep)
    assert apps.creates == []


def test_a_number_the_route_cannot_give_is_refused_with_what_the_route_answered() -> (
    None
):
    apps = fake_with_tiers_and_environment()
    apps.revisions_route_status = 404
    settings = make_settings(
        defaults={"environment_id": "env-1", "environment_revision": 5}
    )
    with pytest.raises(DominoDeployError) as refusal:
        deploy(make_client(apps), settings, "all", launch=LAUNCH, sleep=no_sleep)
    assert "latest (7), active (7) and restricted (none)" in str(refusal.value)
    assert "returned HTTP 404" in str(refusal.value)
    assert apps.creates == []


def test_a_deploy_file_with_ids_only_asks_domino_nothing_new() -> None:
    ids = {"hardware_tier_id": "small-k8s", "environment_revision_id": "rev-7"}
    apps = deployed(defaults={"environment_id": "env-1", **ids})
    assert len(apps.creates) == 3
    assert lookups_of(apps) == []


def test_a_deploy_that_names_no_revision_number_never_calls_the_internal_route() -> (
    None
):
    apps = deployed(defaults={"environment_id": "env-1", "hardware_tier": "cpu-8"})
    assert INTERNAL_CALL not in apps.received
    assert ENVIRONMENT_CALL not in apps.received


def test_deploying_one_app_looks_up_only_what_that_app_names() -> None:
    apps = deployed("embedder", apps={"reranker": {"hardware_tier": "cpu-61"}})
    assert [c.name for c in apps.creates] == ["rag-embedder"]
    assert lookups_of(apps) == []


def test_the_ids_are_found_through_the_fake_given_to_main(run_main: RunMain) -> None:
    apps = fake_with_tiers_and_environment()
    Path("deploy.yaml").write_text(ENV_1 + "  hardware_tier: cpu-8\n")
    code, _, err = run_main(["deploy", "all"], create_fake_domino(apps))
    assert (code, err) == (0, "")
    assert {c.version.hardwareTierId for c in apps.creates} == {"small-k8s"}
