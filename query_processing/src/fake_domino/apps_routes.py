"""The fake's Domino 6.2 beta Apps API routes, under ``/api/apps/beta/apps``.

List, create, add a version, get (each GET moves an instance one status along,
as the state's ``BaseInstanceScheduler`` decides) and list a version's instances,
over the ``FakeDominoApps`` in the app's state. Routes and shapes are Domino's.

Example:
    >>> fake.include_router(_apps_router)  # as create_fake_domino does
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from .api_shapes import (
    AppCreationRequest,
    AppResponse,
    AppVersionCreationRequest,
    AppVersionResponse,
    FailureEnvelopeV1,
    ListAppInstancesResponse,
    ListAppsResponse,
    PaginatedMetadataV1,
)
from .instances import FakeApp
from .state import FAKE_OWNER, FAKE_PROJECT_NAME, FakeDominoApps

APPS_API_PATH = "/api/apps/beta/apps"

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


def _app_response(
    app: FakeApp, request: Request, version: AppVersionResponse
) -> AppResponse:
    fields = app.request.model_dump(exclude_none=True, exclude={"version"})
    project = {"id": app.request.projectId, "name": FAKE_PROJECT_NAME}
    owner = {"ownerId": FAKE_OWNER, "ownerUsername": FAKE_OWNER}
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


def _state(request: Request) -> FakeDominoApps:
    apps: FakeDominoApps = request.app.state.fake_domino_apps
    return apps


FakeState = Annotated[FakeDominoApps, Depends(_state)]


def _find_app(apps: FakeDominoApps, app_id: str) -> FakeApp:
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
    """List the project's Apps; ``name`` matches as a substring.

    Whether Domino matches ``name`` exactly is undocumented.

    Args:
        apps: The fake's state.
        request: The incoming request, whose base URL builds each App's url.
        projectId: Only Apps of this project; None lists every project's.
        name: Only Apps whose name contains this.
        limit: Most Apps in the page.
        offset: Apps skipped before the page starts.

    Returns:
        The page of Apps, with the total count of matches.
    """
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
    """Create an App with its first version.

    Args:
        apps: The fake's state.
        request: The incoming request, whose base URL builds the App's url.
        body: The App and its first version.

    Returns:
        The created App.
    """
    apps.creates.append(body)
    app = apps.add_app(body)
    return _app_response(app, request, apps.current(app))


@_apps_router.post("/{appId}/versions")
def create_app_version(
    apps: FakeState, appId: str, body: AppVersionCreationRequest
) -> AppVersionResponse:
    """Add a version to an App; it becomes the current one.

    Args:
        apps: The fake's state.
        appId: The App that gets the version.
        body: The new version.

    Returns:
        The new version.
    """
    app = _find_app(apps, appId)
    apps.new_versions.append((appId, body))
    apps.add_version(app, body)
    return apps.current(app)


@_apps_router.get("/{appId}")
def get_app(apps: FakeState, request: Request, appId: str) -> AppResponse:
    """Get one App; each GET moves its instance one status along.

    A supplied instance scheduler decides what each GET reports instead.

    Args:
        apps: The fake's state.
        request: The incoming request, whose base URL builds the App's url.
        appId: The App to get.

    Returns:
        The App with its current version.
    """
    app = _find_app(apps, appId)
    return _app_response(app, request, apps.poll(app))


@_apps_router.get("/{appId}/versions/{versionId}/instances")
def list_app_instances(
    apps: FakeState, appId: str, versionId: str
) -> ListAppInstancesResponse:
    """List a version's instances: the latest version's one, once it has started.

    Args:
        apps: The fake's state.
        appId: The App the version belongs to.
        versionId: The version whose instances are listed.

    Returns:
        The instances, in one page.

    Raises:
        HTTPException: 404 if the App has no such version.
    """
    items = apps.instances(_find_app(apps, appId), versionId)
    if items is None:
        raise HTTPException(404, f"no version {versionId} of App {appId}")
    metadata = PaginatedMetadataV1(
        notices=[], requestId="fake-request", limit=len(items), offset=0
    )
    return ListAppInstancesResponse(items=items, metadata=metadata)
