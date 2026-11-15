"""Placement through the Apps API: machines, Queued, statuses, tiers, and the proxy."""

from __future__ import annotations

import logging

import pytest
from test_fake_domino_emulation_support import (
    APPS_PATH,
    CREATE_BODY,
    RUN_TOKEN_AUTH,
    FakeAppLauncher,
    FakeUpstream,
    create,
    http_to,
    placed_apps,
    statuses_seen,
)

from fake_domino.emulation.launcher import AppRunState
from fake_domino.emulation.proxy_settings import FakeDominoProxySettings
from fake_domino.state import FakeDominoApps

TIERS_PATH = "/api/hardwaretiers/v1/hardwaretiers"


def on_tier(tier: str | None) -> dict[str, object]:
    version: dict[str, object] = {"entryScript": "exec bash domino/app.sh"}
    return version if tier is None else {**version, "hardwareTierId": tier}


def test_a_new_app_takes_a_free_machine_of_its_tier_and_starts_there() -> None:
    launcher = FakeAppLauncher()
    http = http_to(placed_apps(launcher, {"RAG_ES_PASSWORD": "s3cret"}))
    created = http.post(APPS_PATH, json={**CREATE_BODY, "version": on_tier("cpu-16")})
    assert created.status_code == 200
    assert created.json()["currentVersion"]["currentInstance"]["status"] == "Preparing"
    [(machine, script)] = launcher.starts
    assert machine == "10.10.0.31"
    assert "export DOMINO_HARDWARE_TIER_ID=cpu-16\n" in script
    assert (
        "export DOMINO_RUN_HOST_PATH=/fake-owner/fake-project/r/notebookSession/app-1-v1-i/\n"
        in script
    )
    assert script.endswith("export RAG_ES_PASSWORD=s3cret\nexec bash domino/app.sh\n")


def test_an_empty_or_missing_tier_is_the_default_one() -> None:
    launcher = FakeAppLauncher()
    http = http_to(placed_apps(launcher))
    create(http, "a", version=on_tier(None))
    create(http, "b", version=on_tier(""))
    assert [machine for machine, _ in launcher.starts] == ["10.10.0.21", "10.10.0.22"]
    assert all("DOMINO_HARDWARE_TIER_ID=cpu-8\n" in s for _, s in launcher.starts)


def test_an_unknown_tier_is_refused_naming_it_and_no_app_is_stored() -> None:
    apps = placed_apps(FakeAppLauncher())
    response = http_to(apps).post(
        APPS_PATH, json={**CREATE_BODY, "version": on_tier("gpu-1")}
    )
    assert response.status_code == 400
    assert response.json() == {
        "errors": [
            "unknown hardwareTierId 'gpu-1'; the emulation has ['cpu-8', 'cpu-16']"
        ],
        "requestId": "fake-request",
    }
    assert apps.apps == {}


def test_a_full_tier_queues_the_app_until_it_gets_a_machine() -> None:
    launcher = FakeAppLauncher()
    http = http_to(placed_apps(launcher))
    create(http, "first", version=on_tier("cpu-16"))
    second = create(http, "second", version=on_tier("cpu-16"))
    assert statuses_seen(http, second, 2) == ["Queued", "Queued"]
    instances = http.get(f"{APPS_PATH}/{second}/versions/app-2-v1/instances").json()
    assert [i["status"] for i in instances["items"]] == ["Queued"]
    assert (
        http.get(f"/apps/{second}/v3/models", headers=RUN_TOKEN_AUTH).status_code == 502
    )
    # A Queued App holds no machine, so a version on another tier may place it.
    http.post(f"{APPS_PATH}/{second}/versions", json=on_tier("cpu-8"))
    assert statuses_seen(http, second, 1) == ["Preparing"]
    assert [machine for machine, _ in launcher.starts] == ["10.10.0.31", "10.10.0.21"]


def test_a_new_version_reuses_the_apps_machine_under_a_new_instance_id() -> None:
    launcher = FakeAppLauncher()
    http = http_to(placed_apps(launcher))
    app_id = create(http, "rag-reranker", version=on_tier("cpu-8"))
    create(http, "other", version=on_tier("cpu-8"))
    posted = http.post(f"{APPS_PATH}/{app_id}/versions", json=on_tier("cpu-8"))
    assert posted.json()["currentInstance"]["id"] == "app-1-v2-i"
    machines = [machine for machine, _ in launcher.starts]
    assert machines == ["10.10.0.21", "10.10.0.22", "10.10.0.21"]
    assert "DOMINO_APP_VERSION_ID=app-1-v2\n" in launcher.starts[-1][1]


def test_a_new_version_on_another_tier_than_its_machine_is_409() -> None:
    http = http_to(placed_apps(FakeAppLauncher()))
    app_id = create(http, "rag-reranker", version=on_tier("cpu-8"))
    posted = http.post(f"{APPS_PATH}/{app_id}/versions", json=on_tier("cpu-16"))
    assert posted.status_code == 409
    assert "runs on a 'cpu-8' machine" in posted.json()["errors"][0]


@pytest.mark.parametrize(
    ("state", "status"),
    [
        (AppRunState.STARTING, "Preparing"),
        (AppRunState.RUNNING, "Running"),
        (AppRunState.FAILED, "Failed"),
    ],
)
def test_the_instance_status_is_the_launchers(state: AppRunState, status: str) -> None:
    launcher = FakeAppLauncher()
    http = http_to(placed_apps(launcher))
    app_id = create(http, "rag-workflow")
    launcher.states["10.10.0.21"] = state
    assert statuses_seen(http, app_id, 1) == [status]
    instances = http.get(f"{APPS_PATH}/{app_id}/versions/app-1-v1/instances").json()
    assert [(i["id"], i["status"]) for i in instances["items"]] == [
        ("app-1-v1-i", status)
    ]


def test_a_start_ssh_cannot_make_is_failed_logged_and_not_a_crash(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="fake_domino")
    launcher = FakeAppLauncher()
    launcher.unreachable.add("10.10.0.21")
    launcher.states["10.10.0.21"] = AppRunState.RUNNING  # an older App still runs
    http = http_to(placed_apps(launcher, {"RAG_ES_PASSWORD": "s3cret"}))
    created = http.post(APPS_PATH, json=CREATE_BODY)
    assert created.status_code == 200
    assert statuses_seen(http, created.json()["id"], 2) == ["Failed", "Failed"]
    assert "Connection timed out" in caplog.text
    assert "s3cret" not in caplog.text


def test_forbidden_still_answers_403_with_placement() -> None:
    apps = FakeDominoApps.from_behaviour(
        "forbidden", scheduler=placed_apps(FakeAppLauncher()).scheduler
    )
    assert http_to(apps).post(APPS_PATH, json=CREATE_BODY).status_code == 403


# --- hardware tiers ---


def test_with_placement_a_tier_reports_its_real_size() -> None:
    http = http_to(placed_apps(FakeAppLauncher()))
    tier = http.get(f"{TIERS_PATH}/cpu-16").json()["hardwareTier"]
    assert (tier["id"], tier["resources"]["cores"]) == ("cpu-16", 16)
    assert tier["resources"]["memory"] == {"value": 64, "units": "GiB"}
    assert tier["flags"]["isDefault"] is False
    assert http.get(f"{TIERS_PATH}/cpu-8").json()["hardwareTier"]["flags"]["isDefault"]


def test_with_placement_an_unknown_tier_is_404() -> None:
    response = http_to(placed_apps(FakeAppLauncher())).get(f"{TIERS_PATH}/gpu-1")
    assert response.status_code == 404
    assert "no hardware tier 'gpu-1'" in response.json()["detail"]


def test_without_placement_any_tier_keeps_its_fixed_size() -> None:
    tier = http_to(FakeDominoApps()).get(f"{TIERS_PATH}/gpu-1").json()["hardwareTier"]
    assert (tier["resources"]["cores"], tier["resources"]["memory"]["value"]) == (4, 16)


# --- the App proxy ---


def test_the_proxy_forwards_a_placed_app_to_its_machine() -> None:
    upstream = FakeUpstream()
    http = http_to(placed_apps(FakeAppLauncher()), None, upstream.transport())
    create(http, "rag-reranker")
    http.get("/apps/app-1/v3/models", headers=RUN_TOKEN_AUTH)
    http.get(
        "/fake-owner/fake-project/r/notebookSession/app-1-v1-i/readyz",
        headers=RUN_TOKEN_AUTH,
    )
    assert [str(r.url) for r in upstream.seen] == [
        "http://10.10.0.21:8888/v3/models",
        "http://10.10.0.21:8888/readyz",
    ]


def test_an_upstream_flag_wins_over_the_apps_machine() -> None:
    upstream = FakeUpstream()
    proxy = FakeDominoProxySettings(upstreams={"rag-reranker": "http://127.0.0.1:9001"})
    http = http_to(placed_apps(FakeAppLauncher()), proxy, upstream.transport())
    create(http, "rag-reranker")
    http.get("/apps/app-1/v3/models", headers=RUN_TOKEN_AUTH)
    assert [str(r.url) for r in upstream.seen] == ["http://127.0.0.1:9001/v3/models"]


def test_the_swagger_description_tells_emulation_from_domino() -> None:
    description = http_to(FakeDominoApps()).app.description  # type: ignore[attr-defined]
    assert "Placement (`--placement FILE`; emulation only" in description
    assert "the core's `--behaviour` scripts are replaced by placement" in description
    assert "an unknown `hardwareTierId` is 400" in description
    assert "Documented by Domino" in description
    assert "Assumed (not in Domino's docs)" in description
