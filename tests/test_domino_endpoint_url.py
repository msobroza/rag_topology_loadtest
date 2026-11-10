"""Tests of ``endpoint_url``: where an App answers, from its running instance."""

from __future__ import annotations

import pytest

from domino_deploy_helpers import (
    APPS_PATH,
    INSTANCES_URL,
    add_app,
    make_client,
)
from rag_load_test.deploy.apps_client import DominoApp, DominoDeployError
from rag_load_test.deploy.endpoint_url import endpoint_url
from rag_load_test.deploy.fake_domino import FakeDominoApps


def found(apps: FakeDominoApps, name: str = "rag-reranker") -> DominoApp:
    app = make_client(apps).find(name)
    assert app is not None
    return app


def test_the_url_is_the_running_instance_under_the_owner_and_project() -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-reranker")
    client = make_client(apps)
    app = found(apps)
    assert endpoint_url(client, app, "") == f"{INSTANCES_URL}/app-1-v1-i"
    path = f"{APPS_PATH}/app-1/versions/app-1-v1/instances"
    assert ("GET", path) in apps.received


def test_domino_host_replaces_the_host_of_the_apps_url() -> None:
    apps = FakeDominoApps(["Running"])
    add_app(apps, "rag-reranker")
    url = endpoint_url(make_client(apps), found(apps), "https://domino.example/")
    assert url == (
        "https://domino.example/fake-owner/fake-project/r/notebookSession/app-1-v1-i"
    )


@pytest.mark.parametrize(
    ("statuses", "seen"), [(["Pending"], "Pending"), ([None], "none")]
)
def test_no_running_instance_names_the_app_and_the_statuses_seen(
    statuses: list[str | None], seen: str
) -> None:
    apps = FakeDominoApps(statuses)
    add_app(apps, "rag-reranker")
    with pytest.raises(DominoDeployError) as caught:
        endpoint_url(make_client(apps), found(apps), "")
    assert str(caught.value) == (
        "App rag-reranker has no Running instance of its current version app-1-v1 "
        f"to call (instance statuses seen: {seen})"
    )


def app_without(**fields: object) -> DominoApp:
    values: dict[str, object] = {
        "id": "app-1",
        "name": "rag-reranker",
        "url": "https://domino.example/apps/app-1",
        "version_id": "app-1-v1",
        "status": None,
        "owner": "alice",
        "project": "rag",
        **fields,
    }
    return DominoApp(**values)  # type: ignore[arg-type]


def test_an_app_with_no_current_version_has_no_url() -> None:
    with pytest.raises(DominoDeployError) as caught:
        endpoint_url(make_client(FakeDominoApps()), app_without(version_id=None), "")
    assert str(caught.value) == (
        "App rag-reranker has no current version, so no instance to call: "
        "deploy it again"
    )


def test_without_a_host_anywhere_it_says_to_set_domino_host() -> None:
    with pytest.raises(DominoDeployError) as caught:
        endpoint_url(make_client(FakeDominoApps()), app_without(url=""), "")
    assert str(caught.value) == (
        "App rag-reranker has no url to take the Domino host from: set "
        "domino_host in deploy.yaml (for example https://domino.example.com)"
    )


@pytest.mark.parametrize(
    ("fields", "missing"),
    [
        ({"owner": "", "project": "rag"}, "project.ownerUsername"),
        ({"owner": "alice", "project": ""}, "project.name"),
        ({"owner": "", "project": ""}, "project.ownerUsername and project.name"),
    ],
)
def test_an_app_without_owner_or_project_has_no_url(
    fields: dict[str, str], missing: str
) -> None:
    with pytest.raises(DominoDeployError) as caught:
        endpoint_url(make_client(FakeDominoApps()), app_without(**fields), "")
    assert str(caught.value) == (
        f"App rag-reranker has no {missing} in the Apps API answer, so the URL "
        "of its instance cannot be built"
    )
