"""Tests of ``DominoAppsClient`` and ``wait_until_running`` over the fake Domino,
driven through ``publish_app``: find, create or add a version, then wait.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel

from domino_deploy_helpers import (
    APPS_PATH,
    APPS_URL,
    BOOTSTRAP_AND_EXEC,
    LAUNCH,
    add_app,
    make_client,
    make_settings,
    no_sleep,
)
from rag_load_test.domino_apps import (
    NO_INSTANCE_GRACE_S,
    WAIT_TIMEOUT_S,
    DominoApp,
    DominoDeployError,
)
from rag_load_test.domino_deploy import (
    DeploySettings,
    publish_app,
)
from rag_load_test.fake_domino import FakeDominoApps


def sent(body: BaseModel) -> dict[str, object]:
    """A recorded request body as rag-deploy sent it (no defaults filled in)."""
    return body.model_dump(exclude_unset=True)


def publish(
    apps: FakeDominoApps,
    settings: DeploySettings,
    part: str,
    sleep: Callable[[float], None] = no_sleep,
) -> DominoApp:
    """``publish_app`` with no model server URLs and no app settings."""
    client = make_client(apps)
    return publish_app(client, settings, part, {}, {}, launch_script=LAUNCH, sleep=sleep)  # type: ignore[arg-type]


def test_publish_creates_the_app_with_the_documented_body() -> None:
    apps = FakeDominoApps(["Running"])
    settings = make_settings(model_server="ovms", ovms_bundle="/mnt/code/ovms")
    app = publish(apps, settings, "reranker")
    assert [sent(body) for body in apps.creates] == [
        {
            "name": "rag-reranker",
            "projectId": "proj-1",
            "visibility": "AUTHENTICATED",
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


def test_create_body_sends_the_hardware_tier_only_when_set() -> None:
    apps = FakeDominoApps(["Running"])
    settings = make_settings(topology="monolith", hardware_tier_id="tier-8")
    publish(apps, settings, "workflow")
    version = sent(apps.creates[0].version)
    assert {k: v for k, v in version.items() if k != "entryScript"} == {
        "environmentId": "env-1",
        "hardwareTierId": "tier-8",
    }


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
    assert gets == [APPS_PATH] + [f"{APPS_PATH}/app-1"] * 3


def test_publish_polls_until_running_and_prints_one_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    apps = FakeDominoApps(["Queued", "Building", "Running"])
    sleeps: list[float] = []
    publish(apps, make_settings(), "embedder", sleep=sleeps.append)
    assert sleeps == [5.0, 5.0, 5.0]
    assert capsys.readouterr().out == f"rag-embedder: Running {APPS_URL}/app-1\n"


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
    assert sum(sleeps) == NO_INSTANCE_GRACE_S  # a grace, not the full ten minutes
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
    assert sum(sleeps) == WAIT_TIMEOUT_S
