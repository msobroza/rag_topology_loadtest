"""The stand-ins (run token, an App instance's /readyz), tiers and OpenAPI."""

from __future__ import annotations

from fake_domino_support import (
    APPS_PATH,
    CREATE_BODY,
    INSTANCE_PATH,
    create,
    http_to,
)

from fake_domino.server import create_fake_domino
from fake_domino.stand_ins import FAKE_RUN_TOKEN, FAKE_TIER_CORES
from fake_domino.state import FakeDominoApps


def test_the_access_token_stand_in_answers_a_fake_run_token() -> None:
    response = http_to(FakeDominoApps()).get("/access-token")
    assert (response.status_code, response.text) == (200, FAKE_RUN_TOKEN)


def test_forbidden_leaves_the_stand_ins_alone() -> None:
    http = http_to(FakeDominoApps.from_behaviour("forbidden"))
    response = http.get("/access-token")
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
    script = "export RAG_TOPOLOGY=monolith"
    http.post(f"{APPS_PATH}/{app_id}/versions", json={"entryScript": script})
    first = http.get(f"{INSTANCE_PATH}/app-1-v1-i/readyz")
    second = http.get(f"{INSTANCE_PATH}/app-1-v2-i/readyz")
    assert "X-RAG-Topology" not in first.headers
    assert second.headers["X-RAG-Topology"] == "monolith"


def test_readyz_stand_in_takes_the_last_export_of_a_name_as_a_shell_would() -> None:
    http = http_to(FakeDominoApps())
    script = "export RAG_TOPOLOGY=monolith\nexport RAG_TOPOLOGY=split-all"
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
