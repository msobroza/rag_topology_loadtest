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

from .domino_apps_reference import (
    AppCreationRequest,
    AppResponse,
    AppVersionCreationRequest,
    AppVersionResponse,
    FailureEnvelopeV1,
    ListAppsResponse,
    PaginatedMetadataV1,
)

APPS_API_PATH = "/api/apps/beta/apps"
FAKE_DOMINO_URL = "http://fake-domino.test"  # the base URL of an in-process fake
FAKE_RUN_TOKEN = "fake-run-token"
# Where a run's token service listens (ovms.DOMINO_TOKEN_URL): served from the same
# port, the fake answers both the Apps API and the token `rag-deploy status` asks for.
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

    def _status(self, version: _FakeVersion, offset: int = 0) -> str | None:
        return self.statuses[min(version.gets + offset, len(self.statuses) - 1)]


_VERSION_DEFAULTS: dict[str, object] = {
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
}


def _version_response(version: _FakeVersion, status: str | None) -> AppVersionResponse:
    instance = None
    if status is not None:
        instance = {"id": f"{version.id}-i", "createdAt": 0, "status": status}
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


@_stand_ins_router.get("/access-token", response_class=PlainTextResponse)
def access_token() -> str:
    """A fake run token, standing in for the token service of a Domino run."""
    return FAKE_RUN_TOKEN


@_stand_ins_router.get("/apps/{appId}/readyz")
def app_readyz(apps: FakeState, appId: str) -> JSONResponse:
    """What the published App would answer, run headers from its entry script."""
    app = _find_app(apps, appId)
    exported = _exported(app.versions[-1].request.entryScript or "")
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
