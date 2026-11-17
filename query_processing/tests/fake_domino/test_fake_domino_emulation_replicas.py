"""Replicas: an App runs on as many machines as its version's minimum replicas."""

from __future__ import annotations

import logging

import pytest
from test_fake_domino_emulation_support import (
    APPS_PATH,
    CREATE_BODY,
    PLACEMENT,
    RUN_TOKEN_AUTH,
    FakeAppLauncher,
    FakeUpstream,
    create,
    http_to,
    placed_apps,
    statuses_seen,
)

from fake_domino.emulation.launcher import AppRunState
from fake_domino.emulation.placement import AppPlacer
from fake_domino.emulation.placement_settings import HardwareTierPool, PlacementSettings
from fake_domino.emulation.proxy_settings import (
    FakeDominoProxySettings,
    ReplicaBalancing,
)
from fake_domino.state import FakeDominoApps

CPU_4 = HardwareTierPool(
    "cpu-4", 4, 16, ("10.10.0.41", "10.10.0.42", "10.10.0.43"), default=True
)
THREE_MACHINES = PlacementSettings(PLACEMENT.ssh, PLACEMENT.app_port, (CPU_4,))


def replicas(minimum: int | None, **spec: object) -> dict[str, object]:
    autoscaling: dict[str, object] = {"enabled": True, **spec}
    if minimum is not None:
        autoscaling["minReplicas"] = minimum
    return {
        "entryScript": "exec bash domino/app.sh",
        "autoscalingSpecification": autoscaling,
    }


def machines_started(launcher: FakeAppLauncher) -> list[str]:
    return [machine for machine, _ in launcher.starts]


def placer_of(apps: FakeDominoApps) -> AppPlacer:
    scheduler = apps.scheduler
    assert isinstance(scheduler, AppPlacer)
    return scheduler


def test_two_replicas_take_two_machines_and_both_get_the_start_script() -> None:
    launcher = FakeAppLauncher()
    apps = placed_apps(launcher, {"QP_ES_PASSWORD": "s3cret"})
    create(http_to(apps), "rag-reranker", version=replicas(2))
    assert machines_started(launcher) == ["10.10.0.21", "10.10.0.22"]
    [first, second] = [script for _, script in launcher.starts]
    assert first == second
    assert "export DOMINO_APP_VERSION_ID=app-1-v1\n" in first
    assert placer_of(apps).upstreams("app-1") == (
        "http://10.10.0.21:8888",
        "http://10.10.0.22:8888",
    )


@pytest.mark.parametrize(
    "version",
    [
        {"entryScript": "exec bash domino/app.sh"},
        {"autoscalingSpecification": {"enabled": False, "minReplicas": 2}},
        {"autoscalingSpecification": {"enabled": True}},
        {"autoscalingSpecification": {"enabled": True, "minReplicas": 0}},
    ],
)
def test_without_an_enabled_minimum_an_app_runs_once(
    version: dict[str, object],
) -> None:
    launcher = FakeAppLauncher()
    create(http_to(placed_apps(launcher)), "rag-reranker", version=version)
    assert machines_started(launcher) == ["10.10.0.21"]


def test_the_maximum_and_the_targets_change_nothing() -> None:
    launcher = FakeAppLauncher()
    http = http_to(placed_apps(launcher))
    spec = {"maxReplicas": 5, "targetCpuAvgUtilizationPct": 50}
    app_id = create(http, "rag-reranker", version=replicas(1, **spec))
    assert machines_started(launcher) == ["10.10.0.21"]
    shown = http.get(f"{APPS_PATH}/{app_id}").json()["currentVersion"]
    assert shown["autoscalingSpecification"]["maxReplicas"] == 5


def test_more_replicas_than_the_tier_has_machines_is_400_and_nothing_is_stored() -> (
    None
):
    launcher = FakeAppLauncher()
    apps = placed_apps(launcher)
    body = {**CREATE_BODY, "version": {**replicas(2), "hardwareTierId": "cpu-16"}}
    response = http_to(apps).post(APPS_PATH, json=body)
    assert response.status_code == 400
    assert response.json()["errors"] == [
        "2 replicas asked, but hardware tier 'cpu-16' has 1 machine(s)"
    ]
    assert apps.apps == {}
    assert launcher.starts == []


def test_too_few_free_machines_queue_the_app_until_a_get_finds_them() -> None:
    launcher = FakeAppLauncher()
    apps = FakeDominoApps(scheduler=AppPlacer(THREE_MACHINES, launcher))
    http = http_to(apps)
    first = create(http, "first", version=replicas(2))
    second = create(http, "second", version=replicas(2))
    assert statuses_seen(http, second, 2) == ["Queued", "Queued"]
    assert placer_of(apps).upstreams(second) == ()
    assert machines_started(launcher) == ["10.10.0.41", "10.10.0.42"]
    http.post(f"{APPS_PATH}/{first}/versions", json=replicas(1))
    assert statuses_seen(http, second, 1) == ["Preparing"]
    assert machines_started(launcher)[-2:] == ["10.10.0.42", "10.10.0.43"]


@pytest.mark.parametrize(
    ("states", "status"),
    [
        ((AppRunState.RUNNING, AppRunState.STARTING), "Preparing"),
        ((AppRunState.RUNNING, AppRunState.RUNNING), "Running"),
        ((AppRunState.RUNNING, AppRunState.FAILED), "Failed"),
    ],
)
def test_the_instance_runs_only_when_every_replica_does(
    states: tuple[AppRunState, AppRunState], status: str
) -> None:
    launcher = FakeAppLauncher()
    http = http_to(placed_apps(launcher))
    app_id = create(http, "rag-reranker", version=replicas(2))
    launcher.states = dict(zip(["10.10.0.21", "10.10.0.22"], states, strict=True))
    assert statuses_seen(http, app_id, 1) == [status]


def test_one_replica_that_cannot_start_fails_the_instance() -> None:
    launcher = FakeAppLauncher()
    launcher.unreachable.add("10.10.0.22")
    launcher.states = {"10.10.0.21": AppRunState.RUNNING}
    http = http_to(placed_apps(launcher))
    app_id = create(http, "rag-reranker", version=replicas(2))
    assert statuses_seen(http, app_id, 2) == ["Failed", "Failed"]


def test_the_instances_list_still_shows_one_instance() -> None:
    http = http_to(placed_apps(FakeAppLauncher()))
    app_id = create(http, "rag-reranker", version=replicas(2))
    instances = http.get(f"{APPS_PATH}/{app_id}/versions/app-1-v1/instances").json()
    assert [i["id"] for i in instances["items"]] == ["app-1-v1-i"]


def test_fewer_replicas_stop_and_free_the_surplus_for_another_app(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="fake_domino")
    launcher = FakeAppLauncher()
    http = http_to(placed_apps(launcher))
    app_id = create(http, "rag-reranker", version=replicas(2))
    other = create(http, "other")
    assert statuses_seen(http, other, 1) == ["Queued"]
    http.post(f"{APPS_PATH}/{app_id}/versions", json=replicas(1))
    assert launcher.stops == ["10.10.0.22"]
    assert machines_started(launcher) == ["10.10.0.21", "10.10.0.22", "10.10.0.21"]
    assert statuses_seen(http, other, 1) == ["Preparing"]
    assert machines_started(launcher)[-1] == "10.10.0.22"
    assert caplog.text.count('"event": "app_placed"') == 3
    assert '"event": "app_released", "app_id": "app-1", "machine": "10.10.0.22"' in (
        caplog.text
    )


def test_more_replicas_take_more_machines_and_restart_every_one() -> None:
    launcher = FakeAppLauncher()
    apps = placed_apps(launcher)
    http = http_to(apps)
    app_id = create(http, "rag-reranker")
    http.post(f"{APPS_PATH}/{app_id}/versions", json=replicas(2))
    assert machines_started(launcher) == ["10.10.0.21", "10.10.0.21", "10.10.0.22"]
    assert "DOMINO_APP_VERSION_ID=app-1-v2\n" in launcher.starts[-1][1]
    assert len(placer_of(apps).upstreams(app_id)) == 2


def test_more_replicas_than_free_machines_queue_and_keep_the_old_version() -> None:
    launcher = FakeAppLauncher()
    apps = FakeDominoApps(scheduler=AppPlacer(THREE_MACHINES, launcher))
    http = http_to(apps)
    app_id = create(http, "rag-reranker")
    other = create(http, "other", version=replicas(2))
    http.post(f"{APPS_PATH}/{app_id}/versions", json=replicas(2))
    assert statuses_seen(http, app_id, 1) == ["Queued"]
    assert machines_started(launcher) == ["10.10.0.41", "10.10.0.42", "10.10.0.43"]
    assert placer_of(apps).upstreams(app_id) == ("http://10.10.0.41:8888",)
    http.post(f"{APPS_PATH}/{other}/versions", json=replicas(1))
    assert statuses_seen(http, app_id, 1) == ["Preparing"]
    assert sorted(machines_started(launcher)[-2:]) == ["10.10.0.41", "10.10.0.43"]


# --- the App proxy ---


def test_round_robin_forwards_successive_requests_to_both_replicas() -> None:
    upstream = FakeUpstream()
    http = http_to(placed_apps(FakeAppLauncher()), None, upstream.transport())
    create(http, "rag-reranker", version=replicas(2))
    for _ in range(3):
        http.get("/apps/app-1/v3/models", headers=RUN_TOKEN_AUTH)
    assert [r.url.host for r in upstream.seen] == [
        "10.10.0.21",
        "10.10.0.22",
        "10.10.0.21",
    ]


def test_connection_random_keeps_one_connection_on_one_replica() -> None:
    upstream = FakeUpstream()
    proxy = FakeDominoProxySettings(
        replica_balancing=ReplicaBalancing.CONNECTION_RANDOM
    )
    http = http_to(placed_apps(FakeAppLauncher()), proxy, upstream.transport())
    create(http, "rag-reranker", version=replicas(2))
    for _ in range(4):
        http.get("/apps/app-1/v3/models", headers=RUN_TOKEN_AUTH)
    assert len({r.url.host for r in upstream.seen}) == 1


def test_an_upstream_flag_still_wins_over_the_replicas() -> None:
    upstream = FakeUpstream()
    proxy = FakeDominoProxySettings(upstreams={"rag-reranker": "http://127.0.0.1:9001"})
    http = http_to(placed_apps(FakeAppLauncher()), proxy, upstream.transport())
    create(http, "rag-reranker", version=replicas(2))
    for _ in range(2):
        http.get("/apps/app-1/v3/models", headers=RUN_TOKEN_AUTH)
    assert {r.url.host for r in upstream.seen} == {"127.0.0.1"}
