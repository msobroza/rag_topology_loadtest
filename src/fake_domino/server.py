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
from collections.abc import Awaitable, Callable, Sequence

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.testclient import TestClient
from starlette.types import ASGIApp

from .api_shapes import FailureEnvelopeV1
from .apps_routes import APPS_API_PATH, _apps_router, _state
from .stand_ins import _stand_ins_router, _tiers_router
from .state import FAKE_DOMINO_BEHAVIOURS, FakeDominoApps

FAKE_DOMINO_URL = "http://fake-domino.test"  # the base URL of an in-process fake
# Where a run's token service listens (run_token_url in deploy.yaml): served from
# the same port, the fake answers both the Apps API and the token `rag-deploy
# status` asks for.
FAKE_DOMINO_PORT = 8899


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
