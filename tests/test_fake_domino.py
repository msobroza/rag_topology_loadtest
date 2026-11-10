"""The fake Domino on its own, through plain HTTP: behaviours, routes, OpenAPI."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fake_domino.server import FAKE_DOMINO_URL, create_fake_domino, main
from fake_domino.stand_ins import FAKE_RUN_TOKEN, FAKE_TIER_CORES
from fake_domino.state import FakeDominoApps

APPS_PATH = "/api/apps/beta/apps"
INSTANCE_PATH = "/fake-owner/fake-project/r/notebookSession"  # + /{instance id}
CREATE_BODY = {
    "name": "rag-workflow",
    "projectId": "proj-1",
    "visibility": "AUTHENTICATED",
    "entryPoint": "domino/app.sh",
    "version": {"entryScript": "exec bash domino/app.sh", "environmentId": "env-1"},
}


def http_to(apps: FakeDominoApps) -> TestClient:
    return TestClient(create_fake_domino(apps), base_url=FAKE_DOMINO_URL)


def statuses_seen(http: TestClient, app_id: str, gets: int) -> list[object]:
    """The instance status of each of ``gets`` GETs of the App (None: no instance)."""
    seen = []
    for _ in range(gets):
        version = http.get(f"{APPS_PATH}/{app_id}").json()["currentVersion"]
        seen.append((version.get("currentInstance") or {}).get("status"))
    return seen


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


def create(http: TestClient, name: str, project_id: str = "proj-1") -> str:
    body = {**CREATE_BODY, "name": name, "projectId": project_id}
    return str(http.post(APPS_PATH, json=body).json()["id"])


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
    with pytest.raises(ValueError) as caught:
        FakeDominoApps.from_behaviour("flaky")
    assert str(caught.value) == (
        "unknown fake Domino behaviour 'flaky'; expected one of: "
        "starts, slow, fails, never-starts, forbidden"
    )


def test_the_access_token_stand_in_answers_a_fake_run_token() -> None:
    response = http_to(FakeDominoApps()).get("/access-token")
    assert (response.status_code, response.text) == (200, FAKE_RUN_TOKEN)


RUN_SCRIPT = (
    "export RAG_ROLE=workflow\n"
    "export RAG_TOPOLOGY=split-all\n"
    "export RAG_RUN_LABEL='split all 8 cores'\n"
    "exec bash domino/app.sh"
)


def test_readyz_stand_in_reports_the_run_headers_from_the_entry_script() -> None:
    http = http_to(FakeDominoApps())
    body = {**CREATE_BODY, "version": {"entryScript": RUN_SCRIPT}}
    assert http.post(APPS_PATH, json=body).json()["id"] == "app-1"
    auth = {"Authorization": f"Bearer {FAKE_RUN_TOKEN}"}
    response = http.get(f"{INSTANCE_PATH}/app-1-v1-i/readyz", headers=auth)
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["checks"][0]["ok"] is True
    assert response.headers["X-RAG-Topology"] == "split-all"
    assert response.headers["X-RAG-Run-Label"] == "split all 8 cores"


def test_readyz_stand_in_has_no_run_headers_when_the_script_has_none() -> None:
    http = http_to(FakeDominoApps())
    create(http, "rag-workflow")
    auth = {"Authorization": f"Bearer {FAKE_RUN_TOKEN}"}
    response = http.get(f"{INSTANCE_PATH}/app-1-v1-i/readyz", headers=auth)
    assert response.status_code == 200
    assert "X-RAG-Topology" not in response.headers
    assert "X-RAG-Run-Label" not in response.headers


def test_readyz_of_an_unknown_instance_is_404() -> None:
    auth = {"Authorization": f"Bearer {FAKE_RUN_TOKEN}"}
    response = http_to(FakeDominoApps()).get(
        f"{INSTANCE_PATH}/app-9-v1-i/readyz", headers=auth
    )
    assert response.status_code == 404


def test_readyz_answers_for_the_version_the_instance_belongs_to() -> None:
    http = http_to(FakeDominoApps())
    app_id = create(http, "rag-workflow")
    script = "export RAG_TOPOLOGY=monolith"
    http.post(f"{APPS_PATH}/{app_id}/versions", json={"entryScript": script})
    first = http.get(f"{INSTANCE_PATH}/app-1-v1-i/readyz")
    second = http.get(f"{INSTANCE_PATH}/app-1-v2-i/readyz")
    assert "X-RAG-Topology" not in first.headers
    assert second.headers["X-RAG-Topology"] == "monolith"


def test_the_openapi_document_shows_the_documented_contract() -> None:
    document = http_to(FakeDominoApps()).get("/openapi.json").json()
    assert {
        f"{APPS_PATH}": {"get", "post"},
        f"{APPS_PATH}/{{appId}}": {"get"},
        f"{APPS_PATH}/{{appId}}/versions": {"post"},
        f"{APPS_PATH}/{{appId}}/versions/{{versionId}}/instances": {"get"},
        "/api/hardwaretiers/v1/hardwaretiers/{hardwareTierId}": {"get"},
    }.items() <= {path: set(ops) for path, ops in document["paths"].items()}.items()
    schemas = document["components"]["schemas"]
    assert sorted(schemas["AppCreationRequest"]["required"]) == [
        "name",
        "projectId",
        "version",
        "visibility",
    ]
    assert "required" not in schemas["AppVersionCreationRequest"]
    assert schemas["AppAutoscalingSpecification"]["required"] == ["enabled"]
    assert sorted(schemas["ListAppInstancesResponse"]["required"]) == [
        "items",
        "metadata",
    ]
    assert "FailureEnvelopeV1" in schemas


def test_the_stand_ins_are_tagged_as_not_the_apps_api() -> None:
    document = http_to(FakeDominoApps()).get("/openapi.json").json()
    tags = {
        path: document["paths"][path]["get"]["tags"]
        for path in (
            "/access-token",
            "/{owner}/{project}/r/notebookSession/{instanceId}/readyz",
        )
    }
    assert all(tag == ["Stand-ins (not Domino's Apps API)"] for tag in tags.values())


def test_the_swagger_description_lists_every_behaviour() -> None:
    description = create_fake_domino().description
    for name in ("starts", "slow", "fails", "never-starts", "forbidden"):
        assert f"`{name}`" in description
    assert "not what a live Domino does" in description


def test_the_command_serves_the_named_behaviour_on_the_run_token_port() -> None:
    served: list[tuple[object, str, int]] = []
    main(
        ["--behaviour", "fails"],
        serve=lambda app, host, port: served.append((app, host, port)),
    )
    [(app, host, port)] = served
    # 8899 is where `rag-deploy status` asks for the run token (run_token_url)
    assert (host, port) == ("127.0.0.1", 8899)
    assert app.state.fake_domino_apps.statuses == ["Pending", "Failed"]  # type: ignore[attr-defined]


def test_the_command_refuses_an_unknown_behaviour(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["--behaviour", "flaky"], serve=lambda app, host, port: None)
    assert exited.value.code == 2
    assert "invalid choice: 'flaky'" in capsys.readouterr().err


def test_an_app_created_without_entry_point_gets_a_placeholder() -> None:
    body = {key: value for key, value in CREATE_BODY.items() if key != "entryPoint"}
    created = http_to(FakeDominoApps()).post(APPS_PATH, json=body)
    assert created.status_code == 200
    assert created.json()["entryPoint"] == "(none: the inline entryScript runs)"


def test_a_visibility_the_reference_does_not_list_is_refused() -> None:
    http = http_to(FakeDominoApps())
    response = http.post(APPS_PATH, json={**CREATE_BODY, "visibility": "EVERYONE"})
    assert response.status_code == 422


def test_forbidden_leaves_the_stand_ins_alone() -> None:
    http = http_to(FakeDominoApps.from_behaviour("forbidden"))
    response = http.get("/access-token")
    assert (response.status_code, response.text) == (200, FAKE_RUN_TOKEN)


def test_readyz_stand_in_takes_the_last_export_of_a_name_as_a_shell_would() -> None:
    http = http_to(FakeDominoApps())
    script = "export RAG_TOPOLOGY=monolith\nexport RAG_TOPOLOGY=split-all"
    body = {**CREATE_BODY, "version": {"entryScript": script}}
    http.post(APPS_PATH, json=body)
    response = http.get(f"{INSTANCE_PATH}/app-1-v1-i/readyz")
    assert response.headers["X-RAG-Topology"] == "split-all"


# --- instances, Agents, autoscaling, hardware tiers ---


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


def test_any_hardware_tier_is_described_with_a_fixed_number_of_cores() -> None:
    http = http_to(FakeDominoApps.from_behaviour("forbidden"))  # not an Apps call
    response = http.get("/api/hardwaretiers/v1/hardwaretiers/cpu-8")
    assert response.status_code == 200
    tier = response.json()["hardwareTier"]
    assert (tier["id"], tier["resources"]["cores"]) == ("cpu-8", FAKE_TIER_CORES)
    assert response.json()["metadata"] == {"notices": [], "requestId": "fake-request"}
