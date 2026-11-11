"""A fake Domino 6.2 (FastAPI): its Apps API, and its App proxy around a local App.

The Apps API is transcribed from the Domino 6.2 Platform API reference, so it
shows what ``rag-deploy`` expects of Domino, not what a live Domino does. The App
proxy (``app_proxy.py``) gates a request, adds the headers Domino documents and
forwards it to an App running locally, so a real app can run "behind Domino" on
a laptop. Used three ways: its ``/docs`` page documents both, ``rag-fake-domino``
serves it locally, and ``rag-deploy --dry-run`` runs the real deploy code
against it in-process.

Example:
    >>> main(["--upstream", "rag-reranker=http://127.0.0.1:9001"])  # then /docs
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
from .app_proxy import FakeAppProxy
from .apps_routes import APPS_API_PATH, _apps_router, _state
from .identity import API_KEY_HEADER, FAKE_API_KEY, FAKE_RUN_TOKEN, FAKE_USERNAME
from .proxy_routes import _identity_router, _proxy_router
from .proxy_settings import FakeDominoProxySettings, UnauthenticatedResponse
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


def create_fake_domino(
    apps: FakeDominoApps | None = None,
    *,
    proxy: FakeDominoProxySettings | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the fake Domino over ``apps``, its App proxy set by ``proxy``.

    Its Swagger description lists every behaviour of ``FAKE_DOMINO_BEHAVIOURS``
    and what of the App proxy is documented by Domino and what is assumed.

    Args:
        apps: The state it serves; None takes a fresh one that ``starts``.
        proxy: The App proxy's options; None takes the defaults (401, identity
            JWT on, prefix stripped, no upstream).
        upstream_transport: Carries the requests forwarded to an App; None
            opens real connections.

    Returns:
        The FastAPI app.

    Example:
        >>> TestClient(create_fake_domino(), base_url=FAKE_DOMINO_URL)
    """
    fake = FastAPI(title="Fake Domino 6.2", description=_description())
    fake.state.fake_domino_apps = apps or FakeDominoApps()
    settings = proxy or FakeDominoProxySettings()
    fake.state.fake_domino_proxy = FakeAppProxy(settings, upstream_transport)
    fake.middleware("http")(_record_and_forbid)
    for router in (_apps_router, _tiers_router, _stand_ins_router, _identity_router):
        fake.include_router(router)
    fake.include_router(_proxy_router)  # last: its routes match the widest paths
    return fake


def _description() -> str:
    behaviours = [
        f"- `{name}`: {b.description}" for name, b in FAKE_DOMINO_BEHAVIOURS.items()
    ]
    return "\n".join(
        [_ABOUT, "", "Behaviours (`--behaviour`):", "", *behaviours, "", _PROXY_ABOUT]
    )


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
    """Return an ``httpx.Client`` whose requests go to ``app`` in-process.

    The requests never reach the network.

    Args:
        app: The ASGI app that answers, e.g. ``create_fake_domino()``.

    Returns:
        A client with base URL ``FAKE_DOMINO_URL``.

    Example:
        >>> DominoAppsClient(fake_domino_http(create_fake_domino()), "proj-1")
    """
    return httpx.Client(base_url=FAKE_DOMINO_URL, transport=_InProcessTransport(app))


_ABOUT = (
    "Transcribed from the Domino 6.2 Platform API reference: it shows what "
    "rag-deploy expects of Domino, not what a live Domino does."
)
_PROXY_ABOUT = f"""App proxy (`/apps/{{app_id}}/...` and \
`/{{owner}}/{{project}}/r/notebookSession/{{instance_id}}/...`), forwarding to \
`--upstream NAME=URL` by App name.

Credentials: `Authorization: Bearer {FAKE_RUN_TOKEN}` (what `/access-token` \
answers) or `{API_KEY_HEADER}: {FAKE_API_KEY}`, both the user `{FAKE_USERNAME}`.

Documented by Domino:

- the proxy checks access before a request reaches the App; Agents \
(`AISYSTEM`) share it;
- the App gets `domino-username: <viewer>`, and with secure identity \
propagation `Authorization: Bearer <RS256 JWT>` (audience `apps`; `sub`, \
`preferred_username`, `email`, `given_name`, `family_name`), keys at \
`/auth/realms/DominoRealm/protocol/openid-connect/certs`;
- with `extendedIdentityPropagationToAppsEnabled` the Bearer value is the \
viewer's own token, and no `domino-username` is passed;
- `/access-token` answers a run's token as plain text.

Assumed (not in Domino's docs):

- a script opens an App URL with the API's credentials;
- no valid credential answers 401 (`--unauthenticated redirect`: 302 to the \
login page), before the App is looked up;
- the caller's `Authorization`, `{API_KEY_HEADER}` and `domino-username` are \
removed;
- the Domino prefix is stripped before the App sees the path (`--keep-prefix` \
keeps it);
- the JWT's `iss` is the realm URL; with an API key, the extended "viewer's \
token" is the run token;
- visibility and `accessStatuses` are stored, never enforced."""


def main(
    argv: Sequence[str] | None = None, *, serve: Callable[..., None] = uvicorn.run
) -> None:
    """Serve the fake: ``rag-fake-domino [--behaviour B] [--upstream NAME=URL]...``.

    Args:
        argv: The command-line arguments; None reads ``sys.argv``.
        serve: Runs the ASGI app on a host and port; tests replace uvicorn.

    Example:
        >>> main(["--upstream", "rag-reranker=http://127.0.0.1:9001"])
    """
    args = _parser().parse_args(argv)
    proxy = FakeDominoProxySettings(
        unauthenticated=UnauthenticatedResponse(args.unauthenticated),
        identity_jwt=args.identity_jwt,
        keep_prefix=args.keep_prefix,
        upstreams=dict(args.upstream),
        forward_timeout_seconds=args.forward_timeout,
    )
    apps = FakeDominoApps.from_behaviour(args.behaviour)
    serve(create_fake_domino(apps, proxy=proxy), host=args.host, port=args.port)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-fake-domino",
        description="Serve a fake Domino 6.2: its Apps API, and its App proxy.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=FAKE_DOMINO_PORT)
    parser.add_argument(
        "--behaviour", default="starts", choices=list(FAKE_DOMINO_BEHAVIOURS)
    )
    proxy = parser.add_argument_group("App proxy")
    _add_forwarding_flags(proxy, FakeDominoProxySettings())
    _add_identity_flags(proxy, FakeDominoProxySettings())
    return parser


def _add_forwarding_flags(
    proxy: argparse._ArgumentGroup, defaults: FakeDominoProxySettings
) -> None:
    # Where a request for an App goes, and what of its path it keeps.
    proxy.add_argument(
        "--upstream",
        action="append",
        default=[],
        type=_upstream,
        metavar="NAME=URL",
        help="where the App named NAME runs (repeatable)",
    )
    proxy.add_argument(
        "--keep-prefix",
        action="store_true",
        help="forward the Domino path prefix to the App instead of stripping it",
    )
    proxy.add_argument(
        "--forward-timeout",
        type=float,
        metavar="SECONDS",
        default=defaults.forward_timeout_seconds,
    )


def _add_identity_flags(
    proxy: argparse._ArgumentGroup, defaults: FakeDominoProxySettings
) -> None:
    # Who gets through, and how the App is told who they are.
    proxy.add_argument(
        "--no-identity-jwt",
        dest="identity_jwt",
        action="store_false",
        help="secure identity propagation off: only domino-username is passed",
    )
    proxy.add_argument(
        "--unauthenticated",
        default=defaults.unauthenticated.value,
        choices=[mode.value for mode in UnauthenticatedResponse],
        help="answer to no credential: 401, or a 302 to the login page",
    )


def _upstream(text: str) -> tuple[str, str]:
    # argparse prints an ArgumentTypeError's message as is, the value included.
    name, _, url = text.partition("=")
    if not name or not url.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError(
            f"--upstream {text!r}: expected NAME=URL, "
            "e.g. rag-reranker=http://127.0.0.1:9001"
        )
    return name, url
