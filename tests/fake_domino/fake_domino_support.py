"""What the fake Domino's tests share: building the fake, creating Apps, credentials."""

from __future__ import annotations

from collections.abc import Mapping

import httpx
from fastapi.testclient import TestClient

from fake_domino.identity import FAKE_API_KEY, FAKE_RUN_TOKEN
from fake_domino.proxy_settings import FakeDominoProxySettings
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
RUN_TOKEN_AUTH = {"Authorization": f"Bearer {FAKE_RUN_TOKEN}"}
API_KEY_AUTH = {"X-Domino-Api-Key": FAKE_API_KEY}


def http_to(
    apps: FakeDominoApps,
    proxy: FakeDominoProxySettings | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> TestClient:
    fake = create_fake_domino(apps, proxy=proxy, upstream_transport=upstream_transport)
    return TestClient(fake, base_url=FAKE_DOMINO_URL, follow_redirects=False)


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


class FakeUpstream:
    """An App running behind the fake proxy: records each request, answers 201."""

    def __init__(self) -> None:
        self.seen: list[httpx.Request] = []

    def answer(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        headers = {"X-Upstream": "fake-app", "Content-Type": "text/plain"}
        return httpx.Response(201, headers=headers, content=b"from the app")

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.answer)
