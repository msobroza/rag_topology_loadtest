"""Domino 6.2's beta Apps REST API, as far as ``rag-deploy`` uses it.

Calls go through ``DOMINO_API_PROXY``, the proxy that authenticates requests made
inside a run, so no credentials are sent. The shapes follow the 6.2 Platform API
reference; they have not been exercised against a live Domino.

Example::

    http = httpx.Client(base_url=os.environ["DOMINO_API_PROXY"])
    DominoAppsClient(http, project_id).find("rag-workflow")
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx

APPS_API_PATH = "/api/apps/beta/apps"
WAIT_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 5.0
FAILED_STATUSES = ("Failed", "Error", "Stopped")
# The 6.2 API has no start call: creating a version is assumed to start an instance.
# One that has still not appeared after this long is reported, not waited out.
NO_INSTANCE_GRACE_S = 60.0


class DominoDeployError(Exception):
    """An expected failure: ``rag-deploy`` shows it without a traceback."""


@dataclass(frozen=True)
class DominoApp:
    """The App fields this tool reads; ``status`` is None when there is no instance."""

    id: str
    name: str
    url: str
    version_id: str | None
    status: str | None


def json_mapping(value: object) -> Mapping[str, object]:
    """``value`` when it is a JSON object, else an empty one."""
    return value if isinstance(value, Mapping) else {}


def json_list(value: object) -> list[object]:
    """``value`` when it is a JSON array, else an empty one."""
    return value if isinstance(value, list) else []


def _app_from_json(body: Mapping[str, object]) -> DominoApp:
    version = json_mapping(body.get("currentVersion"))
    instance = json_mapping(version.get("currentInstance"))
    status = instance.get("status")
    version_id = version.get("id")
    return DominoApp(
        id=str(body.get("id", "")),
        name=str(body.get("name", "")),
        url=str(body.get("url") or "").rstrip("/"),
        version_id=str(version_id) if version_id is not None else None,
        status=str(status) if status is not None else None,
    )


class DominoAppsClient:
    """The Apps API calls for one project: find, create, add a version, get."""

    def __init__(self, http: httpx.Client, project_id: str) -> None:
        self.http, self.project_id = http, project_id

    def __enter__(self) -> DominoAppsClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.http.close()

    def find(self, name: str) -> DominoApp | None:
        params = {"projectId": self.project_id, "name": name}
        items = self._call("GET", APPS_API_PATH, params=params).get("items")
        apps = [_app_from_json(json_mapping(item)) for item in json_list(items)]
        # Exact matching is not documented for ``name``, so match it here.
        return next((app for app in apps if app.name == name), None)

    def create(self, name: str, version: Mapping[str, str]) -> DominoApp:
        # No ``entryPoint``: the version's inline ``entryScript`` takes precedence,
        # and where Domino resolves a launch file for a repo under /repos is unknown.
        body = {
            "name": name,
            "projectId": self.project_id,
            "visibility": "AUTHENTICATED",
            "version": dict(version),
        }
        return _app_from_json(self._call("POST", APPS_API_PATH, json=body))

    def add_version(self, app_id: str, version: Mapping[str, str]) -> str | None:
        path = f"{APPS_API_PATH}/{app_id}/versions"
        version_id = self._call("POST", path, json=dict(version)).get("id")
        return str(version_id) if version_id is not None else None

    def get(self, app_id: str) -> DominoApp:
        return _app_from_json(self._call("GET", f"{APPS_API_PATH}/{app_id}"))

    def _call(self, method: str, path: str, **kwargs: object) -> Mapping[str, object]:
        response = self.http.request(method, path, **kwargs)  # type: ignore[arg-type]
        if response.status_code != 200:
            raise DominoDeployError(_describe_failure(method, path, response))
        return json_mapping(response.json())


def _describe_failure(method: str, path: str, response: httpx.Response) -> str:
    detail = response.text[:200]
    if response.status_code in (401, 403):
        errors = json_list(json_mapping(_json_or_none(response)).get("errors"))
        detail = "; ".join(str(error) for error in errors) or detail
    return (
        f"Domino Apps API {method} {path} returned HTTP {response.status_code}: "
        f"{detail}"
    )


def _json_or_none(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:  # 400/404/500 have no documented body
        return None


def wait_until_running(
    client: DominoAppsClient,
    app: DominoApp,
    version_id: str | None,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> DominoApp:
    """Poll the App until ``version_id`` (None = any version) is Running, or raise.

    Polls are counted rather than timed so tests that pass a no-op ``sleep`` end.

    Example::

        wait_until_running(client, client.get("app-1"), "v2", sleep=lambda _: None)
    """
    last_status = "none yet"
    for poll in range(1, int(WAIT_TIMEOUT_S / POLL_INTERVAL_S) + 1):
        sleep(POLL_INTERVAL_S)
        app = client.get(app.id)
        if version_id is not None and app.version_id != version_id:
            last_status = "the previous version is still current"
            continue  # a redeploy that Domino has not switched to yet
        if app.status == "Running":
            print(f"{app.name}: Running {app.url}")
            return app
        _refuse_dead_instance(app, waited_s=poll * POLL_INTERVAL_S)
        last_status = app.status or "no instance yet"
    raise DominoDeployError(
        f"App {app.name} is not Running after {WAIT_TIMEOUT_S:.0f} s; "
        f"last status: {last_status}"
    )


def _refuse_dead_instance(app: DominoApp, *, waited_s: float) -> None:
    if app.status is None and waited_s >= NO_INSTANCE_GRACE_S:
        raise DominoDeployError(
            f"Domino created App {app.name} but did not start it: start it once from "
            f"the Apps page ({app.url}); the Domino 6.2 Apps API has no start call"
        )
    if app.status in FAILED_STATUSES:
        raise DominoDeployError(
            f"App {app.name} is {app.status}; see its logs at {app.url}"
        )
