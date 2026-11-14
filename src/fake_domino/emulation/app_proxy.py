"""The fake's App proxy: wraps a locally running App the way Domino's proxy does.

Documented by Domino (6.2): a proxy checks access before a request reaches the
App, which needs no sign-in of its own; Agents (``AISYSTEM``) share it. It passes
``domino-username: <viewer>``; with secure identity propagation also
``Authorization: Bearer <JWT>`` (``identity.py``), and the App lives at
``/apps/<app-id>/`` (otherwise ``/{owner}/{project}/r/notebookSession/{run id}/``).
With ``extendedIdentityPropagationToAppsEnabled`` the Bearer value is the
viewer's full-scope token instead, and ``domino-username`` is not passed.

Assumed, as Domino's docs do not say (each switchable in
``FakeDominoProxySettings`` where it matters): a script opens an App URL with
the API's credentials (``identity.caller_of``); no credential answers 401 (or a
302 to the login page); the check comes before the App lookup; the caller's own
``Authorization``, ``X-Domino-Api-Key`` and ``domino-username`` are removed; the
Domino prefix is stripped before the App sees the path; and with an API key the
"viewer's token" under extended propagation is the user's run token.
Visibility and ``accessStatuses`` are stored and echoed, never enforced.

Emulation only: an App placed on a machine is forwarded to
``http://<machine>:<app_port>`` unless ``--upstream`` names it. One pooled
``httpx.AsyncClient`` carries every forwarded request for the fake's lifetime;
``aclose`` (on the app's shutdown) closes it.

Example:
    >>> await FakeAppProxy(FakeDominoProxySettings()).answer(request, caller, target)
"""

from __future__ import annotations

from http import HTTPStatus
from typing import NamedTuple
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import PlainTextResponse, Response
from starlette.datastructures import Headers

from ..instances import FakeApp, FakeVersion
from ..stand_ins import readyz_stand_in
from .identity import (
    API_KEY_HEADER,
    FAKE_RUN_TOKEN,
    KEYCLOAK_REALM_PATH,
    FakeIdentityTokens,
    caller_of,
)
from .proxy_settings import FakeDominoProxySettings, UnauthenticatedResponse

DOMINO_USERNAME_HEADER = "domino-username"
LOGIN_PATH = f"{KEYCLOAK_REALM_PATH}/protocol/openid-connect/auth"  # assumed
_CALLER_HEADERS = {"authorization", API_KEY_HEADER.lower(), DOMINO_USERNAME_HEADER}
# Per-connection headers (RFC 9110 section 7.6.1) and those httpx sets itself.
_HOP_HEADERS = {"host", "content-length", "connection", "keep-alive"}
_HOP_HEADERS |= {"transfer-encoding", "te", "upgrade", "proxy-connection"}
# httpx hands back a decoded body, so its length and encoding no longer hold.
_UNRELAYED = {"content-length", "content-encoding", "transfer-encoding", "connection"}


class AppTarget(NamedTuple):
    """The App a proxied request is for, and the path it asks of it."""

    app: FakeApp
    version: FakeVersion  # whose instance answers
    prefix: str  # the Domino path in front of the App's own, e.g. "/apps/app-1"
    sub_path: str  # the App's own path, without the leading "/"
    placed_upstream: str | None = None  # the App's machine, in the emulation


class FakeAppProxy:
    """Gate, rewrite the headers of, and forward a request for an App.

    Args:
        settings: The options Domino's docs leave open.
        transport: Carries forwarded requests; None opens real connections
            (tests pass an ``httpx.MockTransport``). The pooled client closes
            it in ``aclose``.

    Example:
        >>> FakeAppProxy(FakeDominoProxySettings(upstreams={"rag-reranker": url}))
    """

    def __init__(
        self,
        settings: FakeDominoProxySettings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.tokens = FakeIdentityTokens()
        self._transport = transport
        self._client: httpx.AsyncClient | None = None  # made on first forward

    async def aclose(self) -> None:
        """Close the pooled client, if a request ever opened it (app shutdown)."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def gate(self, request: Request) -> str:
        """Let through only a request with a valid credential.

        Args:
            request: The request for an App URL.

        Returns:
            The caller's username.

        Raises:
            HTTPException: 401 with ``WWW-Authenticate``, or 302 to ``LOGIN_PATH``
                in redirect mode, when the request has no valid credential.
        """
        caller = caller_of(request.headers)
        if caller is not None:
            return caller
        if self.settings.unauthenticated is UnauthenticatedResponse.REDIRECT:
            query = urlencode({"redirect_uri": str(request.url)})
            location = {"Location": f"{LOGIN_PATH}?{query}"}
            raise HTTPException(HTTPStatus.FOUND, "sign in first", headers=location)
        raise HTTPException(
            HTTPStatus.UNAUTHORIZED,
            f"no Domino credential on {request.url.path}: send Authorization: "
            f"Bearer <the run token of /access-token> or {API_KEY_HEADER}",
            headers={"WWW-Authenticate": 'Bearer realm="DominoRealm"'},
        )

    async def answer(
        self, request: Request, caller: str, target: AppTarget
    ) -> Response:
        """Answer an authenticated request as the App behind ``target`` would.

        Args:
            request: The incoming request.
            caller: Its username, from ``gate``.
            target: The App it is for.

        Returns:
            The App's answer when an upstream is configured for it or it runs
            on a machine (``--upstream`` wins); otherwise the ``/readyz``
            stand-in, or 502 for any other path.
        """
        name = target.app.request.name
        upstream = self.settings.upstreams.get(name) or target.placed_upstream
        if upstream is None and target.sub_path == "readyz":
            return readyz_stand_in(target.version)
        if upstream is None:
            return PlainTextResponse(
                f"no upstream configured for App {name!r} "
                f"(rag-fake-domino --upstream {name}=URL)",
                status_code=HTTPStatus.BAD_GATEWAY,
            )
        issuer = f"{str(request.base_url).rstrip('/')}{KEYCLOAK_REALM_PATH}"
        headers = _passed_on(request.headers) + self._identity(caller, target, issuer)
        return await self._forward(request, upstream, target, headers)

    def _identity(
        self, caller: str, target: AppTarget, issuer: str
    ) -> list[tuple[str, str]]:
        # One user, one token: under extended propagation the viewer's own token
        # is the run token, whichever credential the caller used.
        if target.version.request.extendedIdentityPropagationToAppsEnabled:
            return [("authorization", f"Bearer {FAKE_RUN_TOKEN}")]
        added = [(DOMINO_USERNAME_HEADER, caller)]
        if self.settings.identity_jwt:
            token = self.tokens.sign(caller, issuer)
            added.append(("authorization", f"Bearer {token}"))
        return added

    async def _forward(
        self,
        request: Request,
        upstream: str,
        target: AppTarget,
        headers: list[tuple[str, str]],
    ) -> Response:
        url = _upstream_url(upstream, target, request.url.query, self.settings)
        client = self._pooled()
        sent = client.build_request(
            request.method, url, headers=headers, content=await request.body()
        )
        try:
            answer = await client.send(sent)
        except httpx.TransportError as exc:
            return PlainTextResponse(
                f"upstream {upstream} of App {target.app.request.name!r} did not "
                f"answer: {type(exc).__name__}: {exc}",
                status_code=HTTPStatus.BAD_GATEWAY,
            )
        return _relayed(answer)

    def _pooled(self) -> httpx.AsyncClient:
        # One client for the fake's lifetime: a new one per request would open a
        # new connection to the App each time, which the proxy's numbers include.
        if self._client is None:
            timeout = self.settings.forward_timeout_seconds
            self._client = httpx.AsyncClient(transport=self._transport, timeout=timeout)
        return self._client


def _passed_on(headers: Headers) -> list[tuple[str, str]]:
    # Every header the App would see but the caller's credentials and identity.
    dropped = _CALLER_HEADERS | _HOP_HEADERS
    return [(name, value) for name, value in headers.items() if name not in dropped]


def _upstream_url(
    upstream: str, target: AppTarget, query: str, settings: FakeDominoProxySettings
) -> str:
    # Built as a string, not with params=, so the query reaches the App unchanged.
    path = f"/{target.sub_path}"
    if settings.keep_prefix:
        path = f"{target.prefix}{path}"
    url = f"{upstream.rstrip('/')}{path}"
    return f"{url}?{query}" if query else url


def _relayed(answer: httpx.Response) -> Response:
    response = Response(content=answer.content, status_code=answer.status_code)
    for name, value in answer.headers.multi_items():
        if name.lower() not in _UNRELAYED:
            response.headers.append(name, value)
    return response
