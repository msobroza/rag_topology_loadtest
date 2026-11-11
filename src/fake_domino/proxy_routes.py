"""The routes of the fake's App proxy, on both URL shapes, and its signing key.

``/apps/{app_id}/{path}`` (secure identity propagation on) and
``/{owner}/{project}/r/notebookSession/{instance_id}/{path}`` (the legacy shape
``rag-deploy`` builds), every HTTP method and any sub-path, handed to
``FakeAppProxy``; and Keycloak's ``certs`` (JWKS) route, which needs no credential.

Example:
    >>> fake.include_router(_proxy_router)  # as create_fake_domino does
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from .app_proxy import AppTarget, FakeAppProxy
from .apps_routes import FakeState
from .identity import JWKS_PATH
from .state import FakeDominoApps

APP_PATH = "/apps/{app_id}"
LEGACY_INSTANCE_PATH = "/{owner}/{project}/r/notebookSession/{instance_id}"
PROXY_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")

_proxy_router = APIRouter(tags=["App proxy (stand-in for Domino's)"])
_identity_router = APIRouter(tags=["Identity (stand-in for Domino's Keycloak)"])


def _proxy(request: Request) -> FakeAppProxy:
    proxy: FakeAppProxy = request.app.state.fake_domino_proxy
    return proxy


@dataclass(frozen=True)
class _Gated:
    # A request that passed the gate: who sent it, and what answers it.
    caller: str
    apps: FakeDominoApps
    proxy: FakeAppProxy


def _gated(request: Request, apps: FakeState) -> _Gated:
    # A dependency, so the gate runs before the App lookup in the route's body.
    proxy = _proxy(request)
    return _Gated(proxy.gate(request), apps, proxy)


Gated = Annotated[_Gated, Depends(_gated)]


async def app_by_id(request: Request, gated: Gated, app_id: str, path: str) -> Response:
    """Proxy a request for the current version of App ``app_id``.

    Args:
        request: The incoming request.
        gated: Its caller, the fake's state and its App proxy.
        app_id: The App, e.g. ``"app-1"``.
        path: The App's own path, e.g. ``"v3/rerank"``.

    Returns:
        What the App, or the fake standing in for it, answers.

    Raises:
        HTTPException: 404 if there is no such App.
    """
    app = gated.apps.apps.get(app_id)
    if app is None:
        raise HTTPException(HTTPStatus.NOT_FOUND, f"no App {app_id}")
    target = AppTarget(app, app.versions[-1], f"/apps/{app_id}", path)
    return await gated.proxy.answer(request, gated.caller, target)


async def app_by_instance(
    request: Request,
    gated: Gated,
    owner: str,
    project: str,
    instance_id: str,
    path: str,
) -> Response:
    """Proxy a request for App instance ``instance_id``, by its legacy URL.

    Args:
        request: The incoming request.
        gated: Its caller, the fake's state and its App proxy.
        owner: The project owner's username (any is accepted).
        project: The project's name (any is accepted).
        instance_id: The instance, e.g. ``"app-1-v1-i"``.
        path: The App's own path, e.g. ``"readyz"``.

    Returns:
        What the App, or the fake standing in for it, answers.

    Raises:
        HTTPException: 404 if no App version has that instance.
    """
    owned = gated.apps.instance_owner(instance_id)
    if owned is None:
        raise HTTPException(HTTPStatus.NOT_FOUND, f"no App instance {instance_id}")
    prefix = f"/{owner}/{project}/r/notebookSession/{instance_id}"
    target = AppTarget(*owned, prefix, path)
    return await gated.proxy.answer(request, gated.caller, target)


# One route per method: one route with many methods shares an OpenAPI operation id.
for _method in PROXY_METHODS:
    _proxy_router.add_api_route(
        f"{APP_PATH}/{{path:path}}",
        app_by_id,
        methods=[_method],
        name=f"app_by_id_{_method.lower()}",
    )
    _proxy_router.add_api_route(
        f"{LEGACY_INSTANCE_PATH}/{{path:path}}",
        app_by_instance,
        methods=[_method],
        name=f"app_by_instance_{_method.lower()}",
    )


@_identity_router.get(JWKS_PATH)
def identity_keys(request: Request) -> dict[str, list[dict[str, object]]]:
    """Serve the public key the identity JWT verifies with (Keycloak's ``certs``).

    Args:
        request: The incoming request, whose app holds the App proxy.

    Returns:
        A JSON Web Key Set with one RS256 key.
    """
    return _proxy(request).tokens.jwks()
