"""The stand-ins (run token, an App instance's /readyz), tiers and OpenAPI."""

from __future__ import annotations

import pytest
from fake_domino_support import (
    APPS_PATH,
    CREATE_BODY,
    INSTANCE_PATH,
    create,
    http_to,
)

from fake_domino.server import create_fake_domino
from fake_domino.stand_ins import (
    FAKE_RUN_TOKEN,
    FAKE_TIER_CORES,
    fake_environment,
    fake_tier,
)
from fake_domino.state import FakeDominoApps

BUILT_7 = "2026-11-12T09:41:00Z"
BUILT_6 = "2026-11-03T16:02:00Z"
TIERS_PATH = "/api/hardwaretiers/v1/hardwaretiers"
ENVIRONMENT_PATH = "/api/environments/v1/environments"
ENVIRONMENTS_LIST = "/api/environments/beta/environments"
REVISIONS_PATH = "/v4/environments/{}/paginatedRevisionData"
ELEVEN_TIERS = [fake_tier(f"tier-{n}", f"cpu-{n}", n, 4 * n) for n in range(1, 12)]
ENV_1 = fake_environment(
    "env-1", (7, "rev-7", BUILT_7), (6, "rev-6", BUILT_6), name="qp-rag", active=6
)


def test_the_access_token_stand_in_answers_a_fake_run_token() -> None:
    response = http_to(FakeDominoApps()).get("/access-token")
    assert (response.status_code, response.text) == (200, FAKE_RUN_TOKEN)


def test_forbidden_leaves_the_stand_ins_alone() -> None:
    http = http_to(FakeDominoApps.from_behaviour("forbidden"))
    response = http.get("/access-token")
    assert (response.status_code, response.text) == (200, FAKE_RUN_TOKEN)


RUN_SCRIPT = (
    "export QP_ROLE=workflow\n"
    "export QP_TOPOLOGY_ALIAS=split-all\n"
    "export QP_RUN_LABEL='split all 8 cores'\n"
    "exec bash domino/app.sh"
)


def test_readyz_stand_in_reports_the_run_headers_from_the_entry_script() -> None:
    http = http_to(FakeDominoApps())
    body = {**CREATE_BODY, "version": {"entryScript": RUN_SCRIPT}}
    assert http.post(APPS_PATH, json=body).json()["id"] == "app-1"
    response = http.get(f"{INSTANCE_PATH}/app-1-v1-i/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["checks"][0]["ok"] is True
    assert response.headers["X-RAG-Topology"] == "split-all"
    assert response.headers["X-RAG-Run-Label"] == "split all 8 cores"


def test_readyz_stand_in_has_no_run_headers_when_the_script_has_none() -> None:
    http = http_to(FakeDominoApps())
    create(http, "rag-workflow")
    response = http.get(f"{INSTANCE_PATH}/app-1-v1-i/readyz")
    assert response.status_code == 200
    assert "X-RAG-Topology" not in response.headers
    assert "X-RAG-Run-Label" not in response.headers


def test_readyz_of_an_unknown_instance_is_404() -> None:
    response = http_to(FakeDominoApps()).get(f"{INSTANCE_PATH}/app-9-v1-i/readyz")
    assert response.status_code == 404
    assert response.json() == {"detail": "no App instance app-9-v1-i"}


def test_readyz_answers_for_the_version_the_instance_belongs_to() -> None:
    http = http_to(FakeDominoApps())
    app_id = create(http, "rag-workflow")
    script = "export QP_TOPOLOGY_ALIAS=monolith"
    http.post(f"{APPS_PATH}/{app_id}/versions", json={"entryScript": script})
    first = http.get(f"{INSTANCE_PATH}/app-1-v1-i/readyz")
    second = http.get(f"{INSTANCE_PATH}/app-1-v2-i/readyz")
    assert "X-RAG-Topology" not in first.headers
    assert second.headers["X-RAG-Topology"] == "monolith"


def test_readyz_stand_in_takes_the_last_export_of_a_name_as_a_shell_would() -> None:
    http = http_to(FakeDominoApps())
    script = "export QP_TOPOLOGY_ALIAS=monolith\nexport QP_TOPOLOGY_ALIAS=split-all"
    create(http, "rag-workflow", version={"entryScript": script})
    response = http.get(f"{INSTANCE_PATH}/app-1-v1-i/readyz")
    assert response.headers["X-RAG-Topology"] == "split-all"


def test_any_hardware_tier_is_described_with_a_fixed_number_of_cores() -> None:
    http = http_to(FakeDominoApps.from_behaviour("forbidden"))  # not an Apps call
    response = http.get("/api/hardwaretiers/v1/hardwaretiers/cpu-8")
    assert response.status_code == 200
    tier = response.json()["hardwareTier"]
    assert (tier["id"], tier["resources"]["cores"]) == ("cpu-8", FAKE_TIER_CORES)
    assert response.json()["metadata"] == {"notices": [], "requestId": "fake-request"}


def test_the_openapi_document_shows_the_documented_contract() -> None:
    document = http_to(FakeDominoApps()).get("/openapi.json").json()
    assert {
        f"{APPS_PATH}": {"get", "post"},
        f"{APPS_PATH}/{{appId}}": {"get"},
        f"{APPS_PATH}/{{appId}}/versions": {"post"},
        f"{APPS_PATH}/{{appId}}/versions/{{versionId}}/instances": {"get"},
        "/api/hardwaretiers/v1/hardwaretiers/{hardwareTierId}": {"get"},
        TIERS_PATH: {"get"},
        ENVIRONMENTS_LIST: {"get"},
        f"{ENVIRONMENT_PATH}/{{environmentId}}": {"get"},
        REVISIONS_PATH.format("{environmentId}"): {"get"},
        "/{owner}/{project}/r/notebookSession/{instanceId}/readyz": {"get"},
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


def test_the_access_token_is_tagged_as_a_stand_in() -> None:
    document = http_to(FakeDominoApps()).get("/openapi.json").json()
    tags = document["paths"]["/access-token"]["get"]["tags"]
    assert tags == ["Stand-ins (not Domino's Apps API)"]


def test_the_swagger_description_lists_every_behaviour_and_each_assumption() -> None:
    description = create_fake_domino().description
    for name in ("starts", "slow", "fails", "never-starts", "forbidden"):
        assert f"`{name}`" in description
    assert "not what a live Domino does" in description


def test_readyz_needs_no_credential_in_the_core() -> None:
    http = http_to(FakeDominoApps())
    create(http, "rag-workflow")
    assert http.get(f"{INSTANCE_PATH}/app-1-v1-i/readyz").status_code == 200
    assert http.get("/apps/app-1/readyz").status_code == 404  # no proxy here


# --- the lists of hardware tiers and environments, and an environment's revisions ---


def test_the_tiers_are_listed_by_pages_of_ten_with_their_total() -> None:
    http = http_to(FakeDominoApps(tiers=ELEVEN_TIERS))
    first = http.get(TIERS_PATH).json()
    second = http.get(f"{TIERS_PATH}?offset=10").json()
    assert [t["name"] for t in first["hardwareTiers"]] == [
        f"cpu-{n}" for n in range(1, 11)
    ]
    assert first["metadata"] == {
        "notices": [],
        "requestId": "fake-request",
        "limit": 10,
        "offset": 0,
        "totalCount": 11,
    }
    assert [t["id"] for t in second["hardwareTiers"]] == ["tier-11"]


def test_a_listed_tier_gives_its_size_and_its_gpus() -> None:
    tiers = [fake_tier("gpu-k8s", "gpu-1", 8, 32, gpus=1), fake_tier("c", "cpu", 2, 8)]
    listed = http_to(FakeDominoApps(tiers=tiers)).get(TIERS_PATH).json()
    gpu, cpu = listed["hardwareTiers"]
    assert gpu["resources"]["memory"] == {"value": 32.0, "units": "GiB"}
    assert gpu["gpuConfiguration"]["numberOfGpus"] == 1
    assert cpu["gpuConfiguration"] is None


def test_a_fake_started_with_nothing_lists_nothing() -> None:
    http = http_to(FakeDominoApps())
    tiers, environments = (
        http.get(TIERS_PATH).json(),
        http.get(ENVIRONMENTS_LIST).json(),
    )
    assert (tiers["hardwareTiers"], tiers["metadata"]["totalCount"]) == ([], 0)
    assert (environments["environments"], environments["metadata"]["totalCount"]) == (
        [],
        0,
    )


def test_the_environments_are_listed_by_pages() -> None:
    eleven = [fake_environment(f"env-{n}", (1, f"rev-{n}", BUILT_7)) for n in range(11)]
    http = http_to(FakeDominoApps(environments=eleven))
    first = http.get(ENVIRONMENTS_LIST).json()
    second = http.get(f"{ENVIRONMENTS_LIST}?offset=10").json()
    assert [e["name"] for e in first["environments"]] == [f"env-{n}" for n in range(10)]
    assert [e["id"] for e in second["environments"]] == ["env-10"]
    assert first["metadata"]["totalCount"] == 11


def test_an_environment_gives_its_three_revisions() -> None:
    http = http_to(FakeDominoApps(environments=[ENV_1]))
    environment = http.get(f"{ENVIRONMENT_PATH}/env-1").json()["environment"]
    assert (environment["name"], environment["owner"]["username"]) == (
        "qp-rag",
        "fake-owner",
    )
    assert environment["latestRevision"] == {
        "id": "rev-7",
        "number": 7,
        "availableTools": [],
        "status": "succeeded",
    }
    assert environment["selectedRevision"]["number"] == 6
    assert environment["restrictedRevision"] is None


@pytest.mark.parametrize(
    "path", [f"{ENVIRONMENT_PATH}/env-9", REVISIONS_PATH.format("env-9")]
)
def test_an_unknown_environment_is_404(path: str) -> None:
    assert http_to(FakeDominoApps(environments=[ENV_1])).get(path).status_code == 404


def test_every_revision_is_listed_by_the_internal_route_newest_first() -> None:
    http = http_to(FakeDominoApps(environments=[ENV_1]))
    body = http.get(REVISIONS_PATH.format("env-1")).json()
    assert body["revisions"] == [
        {"id": "rev-7", "number": 7, "created": 1794476460000, "active": False},
        {"id": "rev-6", "number": 6, "created": 1793721720000, "active": True},
    ]
    assert body["pageInfo"] == {"totalPages": 1, "currentPage": 0, "pageSize": 5}


def test_the_internal_route_pages_by_five_and_by_the_size_asked() -> None:
    six = [(n, f"rev-{n}", BUILT_7) for n in range(1, 7)]
    http = http_to(FakeDominoApps(environments=[fake_environment("env-1", *six)]))
    by_five = http.get(REVISIONS_PATH.format("env-1")).json()
    by_ten = http.get(REVISIONS_PATH.format("env-1") + "?pageSize=10").json()
    assert [r["number"] for r in by_five["revisions"]] == [6, 5, 4, 3, 2]
    assert by_five["pageInfo"]["totalPages"] == 2
    assert len(by_ten["revisions"]) == 6
    assert by_ten["pageInfo"] == {"totalPages": 1, "currentPage": 0, "pageSize": 10}


def test_the_internal_route_answers_the_status_a_test_sets() -> None:
    apps = FakeDominoApps(environments=[ENV_1])
    apps.revisions_route_status = 404
    assert http_to(apps).get(REVISIONS_PATH.format("env-1")).status_code == 404


@pytest.mark.parametrize(
    "path", [TIERS_PATH, ENVIRONMENTS_LIST, f"{ENVIRONMENT_PATH}/env-1"]
)
def test_forbidden_answers_the_documented_lookups_403(path: str) -> None:
    response = http_to(FakeDominoApps.from_behaviour("forbidden")).get(path)
    assert response.status_code == 403
    assert response.json()["errors"] == [
        "Forbidden by the fake Domino (behaviour: forbidden)"
    ]
