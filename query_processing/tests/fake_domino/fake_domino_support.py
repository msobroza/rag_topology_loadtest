"""What the core fake Domino's tests share: building the fake, creating Apps."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fake_domino.server import FAKE_DOMINO_URL, create_fake_domino
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


def client_of(fake: FastAPI) -> TestClient:
    return TestClient(fake, base_url=FAKE_DOMINO_URL, follow_redirects=False)


def http_to(apps: FakeDominoApps) -> TestClient:
    return client_of(create_fake_domino(apps))


def create(
    http: TestClient,
    name: str,
    project_id: str = "proj-1",
    version: Mapping[str, object] | None = None,
) -> str:
    body: dict[str, object] = {**CREATE_BODY, "name": name, "projectId": project_id}
    if version is not None:
        body["version"] = version
    return str(http.post(APPS_PATH, json=body).json()["id"])


def statuses_seen(http: TestClient, app_id: str, gets: int) -> list[object]:
    """The instance status of each of ``gets`` GETs of the App (None: no instance)."""
    seen = []
    for _ in range(gets):
        version = http.get(f"{APPS_PATH}/{app_id}").json()["currentVersion"]
        seen.append((version.get("currentInstance") or {}).get("status"))
    return seen
