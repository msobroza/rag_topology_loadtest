"""Tests of ``DominoAppsClient`` and ``wait_until_running`` over the fake Domino,
driven through ``publish_app``: find (cached id, then name), create or add a
version, then wait.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from domino_deploy_helpers import (
    AGENT_EXPORTS,
    APPS_PATH,
    APPS_URL,
    BOOTSTRAP_AND_EXEC,
    INSTANCES_URL,
    LAUNCH,
    add_app,
    make_client,
    make_settings,
    no_sleep,
)
from fake_domino.state import FakeDominoApps
from rag_load_test.deploy.app_id_cache import AppIdCache
from rag_load_test.deploy.apps_client import (
    DominoApp,
    DominoAppsClient,
    DominoDeployError,
)
from rag_load_test.deploy.publish import publish_app
from rag_load_test.deploy.settings import DeploySettings


def sent(body: BaseModel) -> dict[str, object]:
    """A recorded request body as rag-deploy sent it (no defaults filled in)."""
    return body.model_dump(exclude_unset=True)


def publish(
    apps: FakeDominoApps,
    settings: DeploySettings,
    part: str,
    sleep: Callable[[float], None] = no_sleep,
    cache: AppIdCache | None = None,
) -> DominoApp:
    """``publish_app`` with no model server URLs."""
    client = make_client(apps, cache=cache)
    return publish_app(client, settings, part, {}, launch_script=LAUNCH, sleep=sleep)  # type: ignore[arg-type]


def test_publish_creates_the_app_with_the_documented_body() -> None:
    apps = FakeDominoApps(["Running"])
    settings = make_settings(model_server="ovms", ovms_bundle="/mnt/code/ovms")
    app = publish(apps, settings, "reranker")
    assert [sent(body) for body in apps.creates] == [
        {
            "name": "rag-reranker",
            "projectId": "proj-1",
            "visibility": "AUTHENTICATED",
            "configurationType": "STANDARD",
            "version": {
                "entryScript": (
                    "export RAG_ROLE=ovms-reranker\n"
                    "export RAG_OVMS_BUNDLE=/mnt/code/ovms\n"
                    f"{BOOTSTRAP_AND_EXEC}"
                ),
                "environmentId": "env-1",
            },
        },
    ]
    assert "entryPoint" not in sent(apps.creates[0])  # the inline script wins
    assert ("POST", APPS_PATH) in apps.received
    assert (app.name, app.status) == ("rag-reranker", "Running")


def test_the_workflow_is_created_as_an_agent() -> None:
    apps = FakeDominoApps(["Running"])
    app = publish(apps, make_settings(topology="monolith"), "workflow")
    assert apps.creates[0].configurationType == "AISYSTEM"
    assert app.configuration_type == "AISYSTEM"  # echoed back by the fake
    assert apps.creates[0].version.entryScript == (
        "export RAG_ROLE=monolith\nexport RAG_TOPOLOGY=monolith\n"
        f"{AGENT_EXPORTS}{BOOTSTRAP_AND_EXEC}"
    )


def test_create_body_sends_the_hardware_tier_and_autoscaling_only_when_set() -> None:
    apps = FakeDominoApps(["Running"])
    workflow = {
        "hardware_tier_id": "tier-8",
        "autoscaling": {"max_replicas": 4, "target_cpu_pct": 70},
    }
    publish(apps, make_settings(topology="monolith", apps={"workflow": workflow}), "workflow")  # fmt: skip
    publish(apps, make_settings(), "embedder")
    workflow_version, embedder_version = (sent(c.version) for c in apps.creates)
    assert {k: v for k, v in workflow_version.items() if k != "entryScript"} == {
        "environmentId": "env-1",
        "hardwareTierId": "tier-8",
        "autoscalingSpecification": {
            "enabled": True,
            "minReplicas": 1,
            "maxReplicas": 4,
            "targetCpuAvgUtilizationPct": 70,
        },
    }
    assert set(embedder_version) == {"entryScript", "environmentId"}


def test_redeploy_posts_a_new_version_instead_of_creating_an_app() -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-reranker")
    publish(apps, make_settings(), "reranker")
    assert apps.creates == []
    assert [(app_id, sent(body)) for app_id, body in apps.new_versions] == [
        (
            "app-1",
            {
                "entryScript": f"export RAG_ROLE=fastapi-reranker\n{BOOTSTRAP_AND_EXEC}",
                "environmentId": "env-1",
            },
        )
    ]
    assert ("POST", f"{APPS_PATH}/app-1/versions") in apps.received


def test_an_existing_app_of_the_other_kind_is_refused_not_converted() -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-workflow", configuration_type="STANDARD")
    with pytest.raises(DominoDeployError) as caught:
        publish(apps, make_settings(topology="monolith"), "workflow")
    assert str(caught.value) == (
        "App rag-workflow exists with configurationType STANDARD (kind app), but "
        "deploy.yaml asks for kind agent (AISYSTEM): delete rag-workflow in Domino "
        "to create it again, or set apps.workflow.kind: app"
    )
    assert (apps.creates, apps.new_versions) == ([], [])


def test_an_app_whose_name_only_contains_the_name_is_not_redeployed() -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-reranker-old")
    add_app(apps, "old-rag-reranker")
    app = publish(apps, make_settings(), "reranker")
    assert [body.name for body in apps.creates] == ["rag-reranker"]
    assert apps.new_versions == []
    assert app.name == "rag-reranker"


def test_redeploy_waits_until_the_new_version_runs() -> None:
    apps = FakeDominoApps(["Running"], stale_gets=2)
    add_app(apps, "rag-reranker")
    app = publish(apps, make_settings(), "reranker")
    gets = [path for method, path in apps.received if method == "GET"]
    assert app.version_id == "app-1-v2"
    assert gets == [APPS_PATH] + [f"{APPS_PATH}/app-1"] * 3 + [
        f"{APPS_PATH}/app-1/versions/app-1-v2/instances"
    ]


def test_publish_polls_until_running_and_prints_where_the_app_answers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    apps = FakeDominoApps(["Queued", "Building", "Running"])
    sleeps: list[float] = []
    publish(apps, make_settings(), "embedder", sleep=sleeps.append)
    assert sleeps == [5.0, 5.0, 5.0]
    out = capsys.readouterr().out
    assert out == f"rag-embedder: Running {INSTANCES_URL}/app-1-v1-i\n"


@pytest.mark.parametrize("status", ["Failed", "Error", "Stopped"])
def test_a_failed_instance_names_the_app_status_and_url(status: str) -> None:
    apps = FakeDominoApps(["Pending", status])
    with pytest.raises(DominoDeployError) as caught:
        publish(apps, make_settings(), "embedder")
    assert str(caught.value) == (
        f"App rag-embedder is {status}; see its logs at {APPS_URL}/app-1"
    )


def test_an_instance_that_appears_after_a_few_polls_is_waited_for() -> None:
    apps = FakeDominoApps([None, None, "Pending", "Running"])
    assert publish(apps, make_settings(), "embedder").status == "Running"


def test_an_app_that_never_gets_an_instance_says_to_start_it_once() -> None:
    apps = FakeDominoApps([None])
    sleeps: list[float] = []
    with pytest.raises(DominoDeployError) as caught:
        publish(apps, make_settings(), "embedder", sleep=sleeps.append)
    assert sum(sleeps) == 60.0  # a grace, not the full ten minutes
    assert str(caught.value) == (
        "Domino created App rag-embedder but did not start it: start it once from "
        f"the Apps page ({APPS_URL}/app-1); the Domino 6.2 Apps API has "
        "no start call"
    )


def test_waiting_times_out_with_the_last_status_seen() -> None:
    apps = FakeDominoApps(["Pulling"])
    sleeps: list[float] = []
    with pytest.raises(DominoDeployError) as caught:
        publish(apps, make_settings(), "embedder", sleep=sleeps.append)
    assert str(caught.value) == (
        "App rag-embedder is not Running after 600 s; last status: Pulling"
    )
    assert sum(sleeps) == 600.0


def test_the_wait_limits_come_from_the_settings() -> None:
    apps = FakeDominoApps(["Pulling"])
    sleeps: list[float] = []
    settings = make_settings(wait_timeout_s=20, poll_interval_s=10)
    with pytest.raises(DominoDeployError) as caught:
        publish(apps, settings, "embedder", sleep=sleeps.append)
    assert sleeps == [10.0, 10.0]
    assert str(caught.value).startswith("App rag-embedder is not Running after 20 s")


def test_an_unreachable_domino_is_reported_with_its_address() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    http = httpx.Client(
        base_url="http://localhost:8899", transport=httpx.MockTransport(refuse)
    )
    with pytest.raises(DominoDeployError) as caught:
        DominoAppsClient(http, "proj-1", AppIdCache(None)).find("rag-workflow")
    assert str(caught.value) == (
        "Domino Apps API GET /api/apps/beta/apps could not be reached at "
        "http://localhost:8899: ConnectError: [Errno 61] Connection refused"
    )


# --- the App id cache: an App is found by id first, and never created twice ---


def test_an_app_created_once_is_redeployed_the_second_time(tmp_path: Path) -> None:
    apps = FakeDominoApps(["Running"])
    cache_file = str(tmp_path / "cache.json")
    for _ in range(2):
        publish(apps, make_settings(), "embedder", cache=AppIdCache(cache_file))
    assert [body.name for body in apps.creates] == ["rag-embedder"]
    assert [app_id for app_id, _ in apps.new_versions] == ["app-1"]
    assert AppIdCache(cache_file).get("proj-1", "rag-embedder") == "app-1"


def test_a_cached_id_is_used_without_listing_by_name() -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-embedder")
    cache = AppIdCache(None)
    cache.remember("proj-1", "rag-embedder", "app-1")
    publish(apps, make_settings(), "embedder", cache=cache)
    assert ("GET", APPS_PATH) not in apps.received
    assert apps.received[0] == ("GET", f"{APPS_PATH}/app-1")
    assert [app_id for app_id, _ in apps.new_versions] == ["app-1"]


def test_a_cached_id_domino_no_longer_has_is_forgotten() -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-embedder")  # app-1
    cache = AppIdCache(None)
    cache.remember("proj-1", "rag-embedder", "app-9")
    publish(apps, make_settings(), "embedder", cache=cache)
    assert apps.received[:2] == [("GET", f"{APPS_PATH}/app-9"), ("GET", APPS_PATH)]
    assert cache.get("proj-1", "rag-embedder") == "app-1"
    assert apps.creates == []


def test_an_app_found_by_name_is_remembered() -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-embedder")
    cache = AppIdCache(None)
    publish(apps, make_settings(), "embedder", cache=cache)
    assert cache.get("proj-1", "rag-embedder") == "app-1"
