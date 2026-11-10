"""Domino 6.2's beta Apps REST API, as far as ``rag-deploy`` uses it.

Calls go through ``DOMINO_API_PROXY``, the proxy that authenticates requests made
inside a run, so no credentials are sent. The shapes follow the 6.2 Platform API
reference; they have not been exercised against a live Domino.

Example::

    http = httpx.Client(base_url=os.environ["DOMINO_API_PROXY"])
    DominoAppsClient(http, project_id, AppIdCache(None)).find("rag-workflow")
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from ..domain.errors import describe_error

if TYPE_CHECKING:  # app_id_cache imports DominoDeployError from here
    from .app_id_cache import AppIdCache

APPS_API_PATH = "/api/apps/beta/apps"
FAILED_STATUSES = ("Failed", "Error", "Stopped")


class DominoDeployError(Exception):
    """An expected failure: ``rag-deploy`` shows it without a traceback."""


@dataclass(frozen=True)
class WaitLimits:
    """How long to wait for an App to run, how often to look, and how long an App
    may go without an instance before it is reported (6.2 has no start call)."""

    timeout_s: float
    poll_interval_s: float
    no_instance_grace_s: float


@dataclass(frozen=True)
class DominoApp:
    """The App fields this tool reads; ``status`` is None when there is no instance."""

    id: str
    name: str
    url: str  # the App's page in Domino, not where it answers (see endpoint_url)
    version_id: str | None
    status: str | None
    configuration_type: str = "STANDARD"
    owner: str = ""  # project.ownerUsername
    project: str = ""  # project.name


@dataclass(frozen=True)
class AppInstance:
    """One item of a version's instance list."""

    id: str
    status: str


def json_mapping(value: object) -> Mapping[str, object]:
    """``value`` when it is a JSON object, else an empty one."""
    return value if isinstance(value, Mapping) else {}


def json_list(value: object) -> list[object]:
    """``value`` when it is a JSON array, else an empty one."""
    return value if isinstance(value, list) else []


def _app_from_json(body: Mapping[str, object]) -> DominoApp:
    version = json_mapping(body.get("currentVersion"))
    instance = json_mapping(version.get("currentInstance"))
    project = json_mapping(body.get("project"))
    status = instance.get("status")
    version_id = version.get("id")
    return DominoApp(
        id=str(body.get("id", "")),
        name=str(body.get("name", "")),
        url=str(body.get("url") or "").rstrip("/"),
        version_id=str(version_id) if version_id is not None else None,
        status=str(status) if status is not None else None,
        # Required by the reference; read leniently like every other field.
        configuration_type=str(body.get("configurationType") or "STANDARD"),
        owner=str(project.get("ownerUsername") or ""),
        project=str(project.get("name") or ""),
    )


class DominoAppsClient:
    """The Apps API calls for one project: find, create, add a version, get.

    ``cache`` remembers each App's id, so an App found or created once is found
    again by id even if the ``name`` filter would not match it exactly.
    """

    def __init__(self, http: httpx.Client, project_id: str, cache: AppIdCache) -> None:
        self.http, self.project_id, self.cache = http, project_id, cache

    def __enter__(self) -> DominoAppsClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.http.close()

    def find(self, name: str) -> DominoApp | None:
        """The App named ``name``: by its cached id first, then by the name filter."""
        app = self._find_cached(name) or self._find_by_name(name)
        if app is not None:
            self.cache.remember(self.project_id, name, app.id)
        return app

    def _find_cached(self, name: str) -> DominoApp | None:
        app_id = self.cache.get(self.project_id, name)
        if app_id is None:
            return None
        response = self._send("GET", f"{APPS_API_PATH}/{app_id}")
        if response.status_code == 404:  # deleted in Domino since
            self.cache.forget(self.project_id, name)
            return None
        return _app_from_json(self._json(response))

    def _find_by_name(self, name: str) -> DominoApp | None:
        params = {"projectId": self.project_id, "name": name}
        items = self._call("GET", APPS_API_PATH, params=params).get("items")
        apps = [_app_from_json(json_mapping(item)) for item in json_list(items)]
        # Exact matching is not documented for ``name``, so match it here.
        return next((app for app in apps if app.name == name), None)

    def create(
        self, name: str, version: Mapping[str, object], configuration_type: str
    ) -> DominoApp:
        # No ``entryPoint``: the version's inline ``entryScript`` takes precedence,
        # and where Domino resolves a launch file for a repo under /repos is unknown.
        body = {
            "name": name,
            "projectId": self.project_id,
            "visibility": "AUTHENTICATED",
            "configurationType": configuration_type,
            "version": dict(version),
        }
        app = _app_from_json(self._call("POST", APPS_API_PATH, json=body))
        self.cache.remember(self.project_id, name, app.id)
        return app

    def add_version(self, app_id: str, version: Mapping[str, object]) -> str | None:
        path = f"{APPS_API_PATH}/{app_id}/versions"
        version_id = self._call("POST", path, json=dict(version)).get("id")
        return str(version_id) if version_id is not None else None

    def get(self, app_id: str) -> DominoApp:
        return _app_from_json(self._call("GET", f"{APPS_API_PATH}/{app_id}"))

    def instances(self, app_id: str, version_id: str) -> list[AppInstance]:
        """The instances of one version of an App, as ``ListAppInstancesResponse``."""
        path = f"{APPS_API_PATH}/{app_id}/versions/{version_id}/instances"
        items = map(json_mapping, json_list(self._call("GET", path).get("items")))
        return [AppInstance(str(i.get("id", "")), str(i.get("status"))) for i in items]

    def _call(self, method: str, path: str, **kwargs: object) -> Mapping[str, object]:
        return self._json(self._send(method, path, **kwargs))

    def _send(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        try:
            response = self.http.request(method, path, **kwargs)  # type: ignore[arg-type]
        except httpx.HTTPError as exc:  # refused, timed out, DNS: no answer at all
            base_url = str(self.http.base_url).rstrip("/")
            raise DominoDeployError(
                f"Domino Apps API {method} {path} could not be reached at "
                f"{base_url}: {describe_error(exc)}"
            ) from exc
        return response

    @staticmethod
    def _json(response: httpx.Response) -> Mapping[str, object]:
        if response.status_code != 200:
            request = response.request
            raise DominoDeployError(
                _describe_failure(request.method, request.url.path, response)
            )
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
    limits: WaitLimits,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> DominoApp:
    """Poll the App until ``version_id`` (None = any version) is Running, or raise.

    Polls are counted rather than timed so tests that pass a no-op ``sleep`` end.

    Example::

        wait_until_running(client, app, "v2", limits, sleep=lambda _: None)
    """
    last_status = "none yet"
    for poll in range(1, int(limits.timeout_s / limits.poll_interval_s) + 1):
        sleep(limits.poll_interval_s)
        app = client.get(app.id)
        if version_id is not None and app.version_id != version_id:
            last_status = "the previous version is still current"
            continue  # a redeploy that Domino has not switched to yet
        if app.status == "Running":
            return app
        waited_s = poll * limits.poll_interval_s
        _refuse_dead_instance(
            app, no_instance_for_too_long=waited_s >= limits.no_instance_grace_s
        )
        last_status = app.status or "no instance yet"
    raise DominoDeployError(
        f"App {app.name} is not Running after {limits.timeout_s:.0f} s; "
        f"last status: {last_status}"
    )


def _refuse_dead_instance(app: DominoApp, *, no_instance_for_too_long: bool) -> None:
    if app.status is None and no_instance_for_too_long:
        raise DominoDeployError(
            f"Domino created App {app.name} but did not start it: start it once from "
            f"the Apps page ({app.url}); the Domino 6.2 Apps API has no start call"
        )
    if app.status in FAILED_STATUSES:
        raise DominoDeployError(
            f"App {app.name} is {app.status}; see its logs at {app.url}"
        )
