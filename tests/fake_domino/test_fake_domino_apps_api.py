"""The fake's Apps API through plain HTTP: behaviours, versions, instances, shapes."""

from __future__ import annotations

import pytest
from fake_domino_support import (
    APPS_PATH,
    CREATE_BODY,
    create,
    http_to,
    statuses_seen,
)
from fastapi.testclient import TestClient

from fake_domino.server import FAKE_DOMINO_URL
from fake_domino.state import FakeDominoApps


@pytest.mark.parametrize(
    ("behaviour", "statuses"),
    [
        ("starts", ["Pending", "Running", "Running"]),
        ("slow", ["Queued", "Pending", "Preparing", "Pulling", "Running", "Running"]),
        ("fails", ["Pending", "Failed", "Failed"]),
        ("never-starts", [None, None, None]),
    ],
)
def test_each_behaviour_scripts_a_new_apps_instance(
    behaviour: str, statuses: list[object]
) -> None:
    http = http_to(FakeDominoApps.from_behaviour(behaviour))
    created = http.post(APPS_PATH, json=CREATE_BODY)
    assert created.status_code == 200
    assert statuses_seen(http, created.json()["id"], len(statuses)) == statuses


def test_listing_matches_the_name_as_a_substring_within_the_project() -> None:
    http = http_to(FakeDominoApps())
    for name in ("rag-reranker", "old-rag-reranker-2", "rag-workflow"):
        create(http, name)
    create(http, "rag-reranker", project_id="other")
    listed = http.get(APPS_PATH, params={"projectId": "proj-1", "name": "reranker"})
    body = listed.json()
    assert [app["name"] for app in body["items"]] == [
        "rag-reranker",
        "old-rag-reranker-2",
    ]
    assert body["metadata"]["totalCount"] == 2
    assert body["items"][0]["url"] == f"{FAKE_DOMINO_URL}/apps/app-1"


def test_listing_pages_with_limit_and_offset() -> None:
    http = http_to(FakeDominoApps())
    for name in ("a", "b", "c"):
        create(http, name)
    body = http.get(APPS_PATH, params={"limit": 1, "offset": 1}).json()
    assert [app["name"] for app in body["items"]] == ["b"]
    assert (body["metadata"]["limit"], body["metadata"]["offset"]) == (1, 1)
    assert body["metadata"]["totalCount"] == 3


def test_a_new_version_becomes_current_and_starts_again() -> None:
    apps = FakeDominoApps.from_behaviour("starts")
    http = http_to(apps)
    app_id = create(http, "rag-workflow")
    statuses_seen(http, app_id, 2)
    posted = http.post(
        f"{APPS_PATH}/{app_id}/versions", json={"entryScript": "exec true"}
    )
    assert (posted.status_code, posted.json()["id"]) == (200, "app-1-v2")
    app = http.get(f"{APPS_PATH}/{app_id}").json()
    assert app["currentVersion"]["id"] == "app-1-v2"
    assert app["currentVersion"]["currentInstance"]["status"] == "Pending"
    assert [app_id for app_id, _ in apps.new_versions] == ["app-1"]


def test_stale_gets_report_the_previous_version_running_after_a_redeploy() -> None:
    http = http_to(FakeDominoApps(["Pending"], stale_gets=2))
    app_id = create(http, "rag-workflow")
    http.post(f"{APPS_PATH}/{app_id}/versions", json={})
    versions = [
        http.get(f"{APPS_PATH}/{app_id}").json()["currentVersion"] for _ in range(3)
    ]
    assert [(v["id"], v["currentInstance"]["status"]) for v in versions] == [
        ("app-1-v1", "Running"),
        ("app-1-v1", "Running"),
        ("app-1-v2", "Pending"),
    ]


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", f"{APPS_PATH}/app-9"), ("POST", f"{APPS_PATH}/app-9/versions")],
)
def test_an_unknown_app_is_404(method: str, path: str) -> None:
    http = http_to(FakeDominoApps())
    assert http.request(method, path, json={}).status_code == 404


def test_a_field_the_reference_does_not_have_is_refused() -> None:
    http = http_to(FakeDominoApps())
    response = http.post(APPS_PATH, json={**CREATE_BODY, "entrypoint": "x"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", APPS_PATH),
        ("POST", APPS_PATH),
        ("GET", f"{APPS_PATH}/app-1"),
        ("POST", f"{APPS_PATH}/app-1/versions"),
    ],
)
def test_forbidden_answers_every_apps_call_with_the_failure_envelope(
    method: str, path: str
) -> None:
    apps = FakeDominoApps.from_behaviour("forbidden")
    response = http_to(apps).request(method, path, json=CREATE_BODY)
    assert response.status_code == 403
    assert response.json() == {
        "errors": ["Forbidden by the fake Domino (behaviour: forbidden)"],
        "requestId": "fake-request",
    }
    assert apps.received == [(method, path)]
    assert apps.creates == []


def test_an_unknown_behaviour_names_the_allowed_ones() -> None:
    with pytest.raises(
        ValueError, match="unknown fake Domino behaviour 'flaky'"
    ) as caught:
        FakeDominoApps.from_behaviour("flaky")
    assert str(caught.value) == (
        "unknown fake Domino behaviour 'flaky'; expected one of: "
        "starts, slow, fails, never-starts, forbidden"
    )


def test_an_app_created_without_entry_point_gets_a_placeholder() -> None:
    body = {key: value for key, value in CREATE_BODY.items() if key != "entryPoint"}
    created = http_to(FakeDominoApps()).post(APPS_PATH, json=body)
    assert created.status_code == 200
    assert created.json()["entryPoint"] == "(none: the inline entryScript runs)"


def test_a_visibility_the_reference_does_not_list_is_refused() -> None:
    http = http_to(FakeDominoApps())
    response = http.post(APPS_PATH, json={**CREATE_BODY, "visibility": "EVERYONE"})
    assert response.status_code == 422


# --- instances, Agents, autoscaling ---


def instances(http: TestClient, version: str = "app-1-v1") -> list[object]:
    path = f"{APPS_PATH}/app-1/versions/{version}/instances"
    body = http.get(path).json()
    return [(item["id"], item["status"]) for item in body["items"]]


def test_an_app_shows_its_owner_and_project() -> None:
    http = http_to(FakeDominoApps())
    project = http.post(APPS_PATH, json=CREATE_BODY).json()["project"]
    assert (project["ownerUsername"], project["name"]) == ("fake-owner", "fake-project")


def test_a_versions_instance_follows_the_status_its_last_get_showed() -> None:
    http = http_to(FakeDominoApps.from_behaviour("starts"))
    create(http, "rag-workflow")
    assert instances(http) == [("app-1-v1-i", "Pending")]  # not polled yet
    statuses_seen(http, "app-1", 2)
    assert instances(http) == [("app-1-v1-i", "Running")]


def test_a_version_with_no_instance_lists_none() -> None:
    http = http_to(FakeDominoApps.from_behaviour("never-starts"))
    create(http, "rag-workflow")
    assert instances(http) == []


def test_after_a_redeploy_only_the_new_version_has_an_instance() -> None:
    http = http_to(FakeDominoApps(["Running"]))
    create(http, "rag-workflow")
    http.post(f"{APPS_PATH}/app-1/versions", json={})
    assert instances(http, "app-1-v1") == []
    assert instances(http, "app-1-v2") == [("app-1-v2-i", "Running")]


@pytest.mark.parametrize("path", ["app-9/versions/v1", "app-1/versions/app-1-v9"])
def test_instances_of_an_unknown_app_or_version_are_404(path: str) -> None:
    http = http_to(FakeDominoApps())
    create(http, "rag-workflow")
    assert http.get(f"{APPS_PATH}/{path}/instances").status_code == 404


def test_configuration_type_and_autoscaling_are_echoed() -> None:
    http = http_to(FakeDominoApps())
    scaling = {"enabled": True, "minReplicas": 1, "maxReplicas": 4}
    version = {**CREATE_BODY["version"], "autoscalingSpecification": scaling}  # type: ignore[dict-item]
    body = {**CREATE_BODY, "configurationType": "AISYSTEM", "version": version}
    app = http.post(APPS_PATH, json=body).json()
    assert app["configurationType"] == "AISYSTEM"
    assert app["currentVersion"]["autoscalingSpecification"] == {
        **scaling,
        "targetCpuAvgUtilizationPct": None,
        "targetMemoryAvgUtilizationPct": None,
        "scaleUpStabilizationWindowSeconds": None,
        "scaleDownStabilizationWindowSeconds": None,
        "useSessionAffinity": None,
    }


def test_an_app_created_without_a_configuration_type_is_standard() -> None:
    app = http_to(FakeDominoApps()).post(APPS_PATH, json=CREATE_BODY).json()
    assert app["configurationType"] == "STANDARD"


def test_autoscaling_without_enabled_is_refused() -> None:
    version = {"autoscalingSpecification": {"maxReplicas": 4}}
    body = {**CREATE_BODY, "version": version}
    assert http_to(FakeDominoApps()).post(APPS_PATH, json=body).status_code == 422
