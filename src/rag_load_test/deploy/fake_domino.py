"""A fake Domino 6.2 Apps API (FastAPI), for trying ``rag-deploy`` without Domino.

Transcribed from the Domino 6.2 Platform API reference, so it shows what
``rag-deploy`` expects of Domino, not what a live Domino does. Used three ways:
its ``/docs`` page documents the API, ``rag-fake-domino`` serves it locally, and
``rag-deploy --dry-run`` runs the real deploy code against it in-process.

Example::

    rag-fake-domino --behaviour slow     # then open http://localhost:8899/docs
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, NamedTuple

import httpx
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.testclient import TestClient
from starlette.types import ASGIApp

from .apps_reference import (
    AppCreationRequest,
    AppInstanceSummaryResponse,
    AppResponse,
    AppVersionCreationRequest,
    AppVersionResponse,
    FailureEnvelopeV1,
    HardwareTierEnvelopeV1,
    ListAppInstancesResponse,
    ListAppsResponse,
    PaginatedMetadataV1,
)

APPS_API_PATH = "/api/apps/beta/apps"
HARDWARE_TIERS_PATH = "/api/hardwaretiers/v1/hardwaretiers"
FAKE_TIER_CORES = 4.0  # every hardware tier the fake describes has this many cores
FAKE_DOMINO_URL = "http://fake-domino.test"  # the base URL of an in-process fake
FAKE_RUN_TOKEN = "fake-run-token"
# Where a run's token service listens (run_token_url in deploy.yaml): served from
# the same port, the fake answers both the Apps API and the token `rag-deploy
# status` asks for.
FAKE_DOMINO_PORT = 8899

# --- Behaviours: what a new version's instance reports, one status per GET ---


class FakeBehaviour(NamedTuple):
    description: str
    statuses: tuple[str | None, ...]  # None = no instance; the last one repeats
    forbidden: bool = False


FAKE_DOMINO_BEHAVIOURS: dict[str, FakeBehaviour] = {
    "starts": FakeBehaviour(
        "a new version's instance is Pending on the first GET, then Running",
        ("Pending", "Running"),
    ),
    "slow": FakeBehaviour(
        "Queued, Pending, Preparing, Pulling, then Running",
        ("Queued", "Pending", "Preparing", "Pulling", "Running"),
    ),
    "fails": FakeBehaviour("Pending, then Failed", ("Pending", "Failed")),
    "never-starts": FakeBehaviour(
        "Domino creates the App or version but no instance (6.2 has no start call)",
        (None,),
    ),
    "forbidden": FakeBehaviour(
        "every Apps API call answers 403 with a FailureEnvelopeV1", (None,), True
    ),
}


@dataclass
class _FakeVersion:
    id: str
    request: AppVersionCreationRequest
    gets: int = 0  # GETs of the App that reported this version


@dataclass
class _FakeApp:
    id: str
    request: AppCreationRequest
    versions: list[_FakeVersion]
    stale_gets_left: int = 0


class FakeDominoApps:
    """The fake's state: its Apps, its scripted statuses and what it received.

    ``statuses``: what a new version's instance reports, one per GET of the App
    (the last repeats; None = no instance). ``stale_gets``: GETs after a new
    version that still report the previous one as current and Running.

    Example::

        create_fake_domino(FakeDominoApps(["Pending", "Running"], stale_gets=2))
    """

    def __init__(
        self,
        statuses: Sequence[str | None] = ("Pending", "Running"),
        *,
        stale_gets: int = 0,
        forbidden: bool = False,
    ) -> None:
        self.statuses, self.stale_gets = list(statuses), stale_gets
        self.forbidden = forbidden
        self.apps: dict[str, _FakeApp] = {}
        self.received: list[tuple[str, str]] = []  # (method, path), in order
        self.creates: list[AppCreationRequest] = []
        self.new_versions: list[tuple[str, AppVersionCreationRequest]] = []

    @classmethod
    def from_behaviour(cls, name: str) -> FakeDominoApps:
        """The state for one of ``FAKE_DOMINO_BEHAVIOURS``, e.g. ``"fails"``."""
        if name not in FAKE_DOMINO_BEHAVIOURS:
            raise ValueError(
                f"unknown fake Domino behaviour {name!r}; expected one of: "
                f"{', '.join(FAKE_DOMINO_BEHAVIOURS)}"
            )
        behaviour = FAKE_DOMINO_BEHAVIOURS[name]
        return cls(behaviour.statuses, forbidden=behaviour.forbidden)

    def add_app(self, request: AppCreationRequest) -> _FakeApp:
        """Store an App with its first version, e.g. to set up a test."""
        app_id = f"app-{len(self.apps) + 1}"
        version = _FakeVersion(f"{app_id}-v1", request.version)
        self.apps[app_id] = _FakeApp(app_id, request, [version])
        return self.apps[app_id]

    def add_version(self, app: _FakeApp, request: AppVersionCreationRequest) -> None:
        app.versions.append(_FakeVersion(f"{app.id}-v{len(app.versions) + 1}", request))
        app.stale_gets_left = self.stale_gets

    def current(self, app: _FakeApp) -> AppVersionResponse:
        """``app``'s current version as a list or a create shows it (not a GET)."""
        version = app.versions[-1]
        return _version_response(version, self._status(version))

    def poll(self, app: _FakeApp) -> AppVersionResponse:
        """What one GET of ``app`` reports as its current version."""
        if app.stale_gets_left:  # a redeploy Domino has not switched to yet
            app.stale_gets_left -= 1
            return _version_response(app.versions[-2], "Running")
        version = app.versions[-1]
        version.gets += 1
        return _version_response(version, self._status(version, offset=-1))

    def instances(
        self, app: _FakeApp, version_id: str
    ) -> list[AppInstanceSummaryResponse] | None:
        """A version's instances (None: no such version), as its last GET showed.

        Only the latest version has one: what Domino does with the previous
        version's instance after a redeploy is not documented.
        """
        if version_id not in (version.id for version in app.versions):
            return None
        version = app.versions[-1]
        status = self._status(version, offset=-1 if version.gets else 0)
        if version.id != version_id or status is None:
            return []
        return [_instance(version, status)]

    def version_of_instance(self, instance_id: str) -> _FakeVersion | None:
        """The version whose instance is ``instance_id``, e.g. ``"app-1-v1-i"``."""
        versions = (v for app in self.apps.values() for v in app.versions)
        return next((v for v in versions if _instance_id(v) == instance_id), None)

    def _status(self, version: _FakeVersion, offset: int = 0) -> str | None:
        return self.statuses[min(version.gets + offset, len(self.statuses) - 1)]


_VERSION_DEFAULTS: dict[str, object] = {
    "extendedIdentityPropagationToAppsEnabled": False,
    "environmentId": "",
    "environmentRevisionId": "",
    "hardwareTierId": "",
    "externalVolumeMountIds": [],
    "netAppVolumeIds": [],
    "tags": [],
}
_APP_DEFAULTS: dict[str, object] = {
    "entryPoint": "(none: the inline entryScript runs)",  # required in a response
    "configurationType": "STANDARD",
    "discoverable": False,
    "mountDatasets": False,
    "mountNetAppVolumes": False,
    "renderIFrame": False,
    "views": 0,
    "accessStatuses": [],
}


def _instance_id(version: _FakeVersion) -> str:
    return f"{version.id}-i"


def _instance(version: _FakeVersion, status: str) -> AppInstanceSummaryResponse:
    return AppInstanceSummaryResponse(
        id=_instance_id(version), createdAt=0, status=status
    )


def _version_response(version: _FakeVersion, status: str | None) -> AppVersionResponse:
    instance = _instance(version, status) if status is not None else None
    fields = version.request.model_dump(exclude_none=True)
    return AppVersionResponse.model_validate(
        {**_VERSION_DEFAULTS, **fields, "id": version.id, "createdAt": 0}
        | {"updatedAt": 0, "currentInstance": instance}
    )


def _app_response(
    app: _FakeApp, request: Request, version: AppVersionResponse
) -> AppResponse:
    fields = app.request.model_dump(exclude_none=True, exclude={"version"})
    project = {"id": app.request.projectId, "name": "fake-project"}
    owner = {"ownerId": "fake-owner", "ownerUsername": "fake-owner"}
    url = f"{str(request.base_url).rstrip('/')}/apps/{app.id}"
    return AppResponse.model_validate(
        {**_APP_DEFAULTS, **fields, "id": app.id, "url": url}
        | {"project": {**project, **owner, "projectType": "Analytic"}}
        | {"currentVersion": version}
    )


# --- The routes ---

_FAILURES: dict[int | str, dict[str, object]] = {
    401: {"model": FailureEnvelopeV1},
    403: {"model": FailureEnvelopeV1},
}
_apps_router = APIRouter(
    prefix=APPS_API_PATH, tags=["Apps (Domino 6.2 beta API)"], responses=_FAILURES
)
_tiers_router = APIRouter(
    prefix=HARDWARE_TIERS_PATH, tags=["Hardware tiers (Domino 6.2 API)"]
)
_stand_ins_router = APIRouter(tags=["Stand-ins (not Domino's Apps API)"])


def _state(request: Request) -> FakeDominoApps:
    apps: FakeDominoApps = request.app.state.fake_domino_apps
    return apps


FakeState = Annotated[FakeDominoApps, Depends(_state)]


def _find_app(apps: FakeDominoApps, app_id: str) -> _FakeApp:
    if app_id not in apps.apps:
        raise HTTPException(404, f"no App {app_id}")
    return apps.apps[app_id]


@_apps_router.get("")
def list_apps(
    apps: FakeState,
    request: Request,
    projectId: str | None = None,
    name: str | None = None,
    limit: int = 100,  # the reference documents no default
    offset: int = 0,
) -> ListAppsResponse:
    """The project's Apps; ``name`` matches as a substring (exactness undocumented)."""
    found = [
        app
        for app in apps.apps.values()
        if projectId in (None, app.request.projectId)
        and (name or "") in app.request.name
    ]
    page = found[offset : offset + limit]
    return ListAppsResponse(
        items=[_app_response(app, request, apps.current(app)) for app in page],
        metadata=PaginatedMetadataV1(
            notices=[],
            requestId="fake-request",
            limit=limit,
            offset=offset,
            totalCount=len(found),
        ),
    )


@_apps_router.post("")
def create_app(
    apps: FakeState, request: Request, body: AppCreationRequest
) -> AppResponse:
    """Create an App with its first version."""
    apps.creates.append(body)
    app = apps.add_app(body)
    return _app_response(app, request, apps.current(app))


@_apps_router.post("/{appId}/versions")
def create_app_version(
    apps: FakeState, appId: str, body: AppVersionCreationRequest
) -> AppVersionResponse:
    """Add a version to an App; it becomes the current one."""
    app = _find_app(apps, appId)
    apps.new_versions.append((appId, body))
    apps.add_version(app, body)
    return apps.current(app)


@_apps_router.get("/{appId}")
def get_app(apps: FakeState, request: Request, appId: str) -> AppResponse:
    """One App; each GET moves its instance one status along."""
    app = _find_app(apps, appId)
    return _app_response(app, request, apps.poll(app))


@_apps_router.get("/{appId}/versions/{versionId}/instances")
def list_app_instances(
    apps: FakeState, appId: str, versionId: str
) -> ListAppInstancesResponse:
    """A version's instances: the latest version's one, once it has started."""
    items = apps.instances(_find_app(apps, appId), versionId)
    if items is None:
        raise HTTPException(404, f"no version {versionId} of App {appId}")
    metadata = PaginatedMetadataV1(
        notices=[], requestId="fake-request", limit=len(items), offset=0
    )
    return ListAppInstancesResponse(items=items, metadata=metadata)


@_tiers_router.get("/{hardwareTierId}")
def get_hardware_tier(hardwareTierId: str) -> HardwareTierEnvelopeV1:
    """Any id is a tier with ``FAKE_TIER_CORES`` cores."""
    memory = {"value": 16, "units": "GiB"}
    resources = {"cores": FAKE_TIER_CORES, "memory": memory}
    flags = dict.fromkeys(_TIER_FLAGS, False)
    tier = {"id": hardwareTierId, "name": hardwareTierId, "flags": flags}
    return HardwareTierEnvelopeV1.model_validate(
        {"hardwareTier": tier | _TIER_DEFAULTS | {"resources": resources | _SHARED}}
        | {"metadata": {"notices": [], "requestId": "fake-request"}}
    )


_TIER_FLAGS = ("isDefault", "isVisible", "isGlobal", "isArchived", "isDataAnalystTier")
_SHARED = {"allowSharedMemoryToExceedDefault": False}
_TIER_DEFAULTS: dict[str, object] = {
    "centsPerMinute": 0.0,
    "creationTime": "2026-01-01T00:00:00Z",
    "updateTime": "2026-01-01T00:00:00Z",
    "podCustomization": {},
    "metadata": {},
}


@_stand_ins_router.get("/access-token", response_class=PlainTextResponse)
def access_token() -> str:
    """A fake run token, standing in for the token service of a Domino run."""
    return FAKE_RUN_TOKEN


@_stand_ins_router.get("/{owner}/{project}/r/notebookSession/{instanceId}/readyz")
def app_readyz(apps: FakeState, instanceId: str) -> JSONResponse:
    """What an App instance would answer, run headers from its entry script."""
    version = apps.version_of_instance(instanceId)
    if version is None:
        raise HTTPException(404, f"no App instance {instanceId}")
    exported = _exported(version.request.entryScript or "")
    headers = {key: exported[name] for key, name in _RUN_HEADERS if name in exported}
    return JSONResponse({"status": "ready", "checks": [_FAKE_CHECK]}, headers=headers)


_RUN_HEADERS = [
    ("X-RAG-Topology", "RAG_TOPOLOGY"),
    ("X-RAG-Run-Label", "RAG_RUN_LABEL"),
]
_FAKE_CHECK = {"name": "fake_domino", "ok": True, "detail": "answered by the fake"}


def _exported(script: str) -> dict[str, str]:
    # The last export of a name wins, as in the shell.
    lines = [shlex.split(line) for line in script.splitlines()]
    pairs = [line[1].partition("=") for line in lines if line[:1] == ["export"]]
    return {name: value for name, _, value in pairs}


async def _record_and_forbid(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    apps = _state(request)
    apps.received.append((request.method, request.url.path))
    if apps.forbidden and request.url.path.startswith(APPS_API_PATH):
        errors = ["Forbidden by the fake Domino (behaviour: forbidden)"]
        body = FailureEnvelopeV1(errors=errors, requestId="fake-request")
        return JSONResponse(body.model_dump(), status_code=403)
    return await call_next(request)


def create_fake_domino(apps: FakeDominoApps | None = None) -> FastAPI:
    """The fake Domino over ``apps`` (default: a fresh one that ``starts``).

    Example::

        TestClient(create_fake_domino(), base_url=FAKE_DOMINO_URL)
    """
    behaviours = [
        f"- `{name}`: {b.description}" for name, b in FAKE_DOMINO_BEHAVIOURS.items()
    ]
    description = "\n".join(
        [_ABOUT, "", "Behaviours (`--behaviour`):", "", *behaviours]
    )
    fake = FastAPI(title="Fake Domino 6.2 Apps API", description=description)
    fake.state.fake_domino_apps = apps or FakeDominoApps()
    fake.middleware("http")(_record_and_forbid)
    fake.include_router(_apps_router)
    fake.include_router(_tiers_router)
    fake.include_router(_stand_ins_router)
    return fake


class _InProcessTransport(httpx.BaseTransport):
    # Starlette's TestClient is built on ``httpx2``, not ``httpx``: wrapping it keeps
    # rag-deploy on a real httpx.Client (its types and its httpx.HTTPError handling).
    def __init__(self, app: ASGIApp) -> None:
        self._test_client = TestClient(app)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        answer = self._test_client.request(
            request.method,
            str(request.url),
            headers=request.headers.multi_items(),
            content=request.read(),
        )
        headers = answer.headers.multi_items()
        return httpx.Response(
            answer.status_code, headers=headers, content=answer.content
        )

    def close(self) -> None:
        self._test_client.close()


def fake_domino_http(app: ASGIApp) -> httpx.Client:
    """An ``httpx.Client`` whose requests go to ``app`` in-process, never the network.

    Example::

        DominoAppsClient(fake_domino_http(create_fake_domino()), "proj-1")
    """
    return httpx.Client(base_url=FAKE_DOMINO_URL, transport=_InProcessTransport(app))


_ABOUT = (
    "Transcribed from the Domino 6.2 Platform API reference: it shows what "
    "rag-deploy expects of Domino, not what a live Domino does."
)


def main(
    argv: Sequence[str] | None = None, *, serve: Callable[..., None] = uvicorn.run
) -> None:
    """``rag-fake-domino [--host H] [--port P] [--behaviour B]``: serve the fake.

    Example::

        main(["--behaviour", "slow"])   # Swagger at http://localhost:8899/docs
    """
    parser = argparse.ArgumentParser(
        prog="rag-fake-domino", description="Serve a fake Domino 6.2 Apps API."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=FAKE_DOMINO_PORT)
    parser.add_argument(
        "--behaviour", default="starts", choices=list(FAKE_DOMINO_BEHAVIOURS)
    )
    args = parser.parse_args(argv)
    apps = FakeDominoApps.from_behaviour(args.behaviour)
    serve(create_fake_domino(apps), host=args.host, port=args.port)
