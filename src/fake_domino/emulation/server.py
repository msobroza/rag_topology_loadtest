"""The Domino emulation: the core fake Domino plus its App proxy and placement.

Emulation only, and removable as one folder: it builds on the core's
``create_fake_domino`` (routes served first, a lifespan, its Swagger text) and
plugs ``AppPlacer`` into the core's ``InstanceScheduler`` seam. In front of each
App URL the proxy checks credentials and rewrites headers as Domino's does;
with ``--placement FILE`` each App runs on a free machine of its hardware tier,
started over SSH. It stays one process: its Apps live in memory.

Example:
    >>> main(["--placement", "/etc/fake-domino/placement.json"])  # then /docs
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import APIRouter, FastAPI

from ..server import create_fake_domino
from ..state import FakeDominoApps
from .about import EMULATION_ABOUT
from .app_proxy import FakeAppProxy
from .flags import emulation_parser
from .placement import AppPlacer
from .placement_settings import load_placement
from .project_env import load_project_env
from .proxy_routes import _identity_router, _proxy_router
from .proxy_settings import FakeDominoProxySettings, UnauthenticatedResponse
from .ssh_launcher import SshAppLauncher
from .tiers import _placed_tiers_router


@asynccontextmanager
async def _lifespan(fake: FastAPI) -> AsyncGenerator[None]:
    # The App proxy's pooled client lives as long as the app.
    yield
    await fake.state.fake_domino_proxy.aclose()


def create_domino_emulation(
    apps: FakeDominoApps | None = None,
    *,
    proxy: FakeDominoProxySettings | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the core fake Domino with the emulation's App proxy and placement.

    Args:
        apps: The state it serves; None takes a fresh one that ``starts``. With
            ``FakeDominoApps(scheduler=AppPlacer(...))`` each App runs on a
            machine, and the hardware tiers route reports the placement's tiers.
        proxy: The App proxy's options; None takes the defaults (401, identity
            JWT on, prefix stripped, no upstream).
        upstream_transport: Carries the requests forwarded to an App; None
            opens real connections.

    Returns:
        The FastAPI app.

    Example:
        >>> TestClient(create_domino_emulation(), base_url=FAKE_DOMINO_URL)
    """
    apps = apps or FakeDominoApps()
    routers: list[APIRouter] = [_identity_router, _proxy_router]
    if isinstance(apps.scheduler, AppPlacer):
        routers.append(_placed_tiers_router)
    fake = create_fake_domino(
        apps, routers=routers, lifespan=_lifespan, about=EMULATION_ABOUT
    )
    settings = proxy or FakeDominoProxySettings()
    fake.state.fake_domino_proxy = FakeAppProxy(settings, upstream_transport)
    return fake


def main(
    argv: Sequence[str] | None = None, *, serve: Callable[..., None] = uvicorn.run
) -> None:
    """Serve the emulation: ``rag-domino-emulation [--placement FILE] ...``.

    Args:
        argv: The command-line arguments; None reads ``sys.argv``.
        serve: Runs the ASGI app on a host and port; tests replace uvicorn.

    Example:
        >>> main(["--placement", "placement.json", "--project-env", "project.env"])
    """
    parser = emulation_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    proxy = FakeDominoProxySettings(
        unauthenticated=UnauthenticatedResponse(args.unauthenticated),
        identity_jwt=args.identity_jwt,
        keep_prefix=args.keep_prefix,
        upstreams=dict(args.upstream),
        forward_timeout_seconds=args.forward_timeout,
    )
    placer = _placer(parser, args.placement, args.project_env)
    apps = FakeDominoApps(scheduler=placer)
    app = create_domino_emulation(apps, proxy=proxy)
    serve(app, host=args.host, port=args.port)


def _placer(
    parser: argparse.ArgumentParser, placement: Path | None, project_env: Path | None
) -> AppPlacer | None:
    # A bad file ends the command with its message, as a bad flag would.
    if placement is None and project_env is not None:
        parser.error("--project-env needs --placement: only placed Apps get it")
    if placement is None:
        return None
    try:
        settings = load_placement(placement)
        variables = load_project_env(project_env) if project_env else {}
    except ValueError as exc:
        parser.error(str(exc))
    return AppPlacer(settings, SshAppLauncher(settings), variables)
