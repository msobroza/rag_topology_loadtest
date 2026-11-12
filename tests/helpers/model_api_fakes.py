"""A scripted HTTP route and builders for the ``adapters.model_api`` tests.

Every request is recorded (method, URL, JSON body, Authorization header, read
timeout) and answered from a script, so auth, batching, ordering and the retry
policy are checked without a model server or a Domino token endpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from rag_load_test.adapters.http.http_policy import HttpPolicy
from rag_load_test.adapters.model_api.api_endpoint_client import ApiEndpointClient
from rag_load_test.adapters.model_api.bearer_token import BearerToken
from rag_load_test.domain.api_endpoint import ApiEndpoint
from rag_load_test.domain.passages import ScoredPassage

TOKEN_URL = "http://token.test/access-token"
TIMEOUT_S = 12.5
NO_WAIT = HttpPolicy(timeout_s=TIMEOUT_S, retries=2, backoff_s=0.0)


@dataclass
class RecordedRequest:
    method: str
    url: str
    json: Any
    authorization: str | None
    read_timeout: float | None


class ScriptedRoute:
    """MockTransport handler: replies (or raises) in script order; last entry repeats."""

    def __init__(self, *script: httpx.Response | Exception) -> None:
        self._script = list(script)
        self.requests: list[RecordedRequest] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        timeout = request.extensions.get("timeout", {}).get("read")
        authorization = request.headers.get("Authorization")
        self.requests.append(
            RecordedRequest(
                request.method, str(request.url), body, authorization, timeout
            )
        )
        outcome = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def methods(self) -> list[str]:
        return [r.method for r in self.requests]


class HttpClients:
    """Opens ``httpx.AsyncClient`` s over scripted routes; closes them all at the end."""

    def __init__(self) -> None:
        self._clients: list[httpx.AsyncClient] = []

    def over(self, route: ScriptedRoute) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(route))
        self._clients.append(client)
        return client

    async def aclose(self) -> None:
        for client in self._clients:
            await client.aclose()


def bearer(
    client: httpx.AsyncClient, api_key: str, *, ttl_s: float = 240.0
) -> BearerToken:
    return BearerToken(
        client, api_key, token_url=TOKEN_URL, ttl_s=ttl_s, policy=NO_WAIT
    )


def endpoint_client(
    client: httpx.AsyncClient,
    endpoint: ApiEndpoint,
    *,
    component: str,
    policy: HttpPolicy = NO_WAIT,  # retries must not sleep in tests
) -> ApiEndpointClient:
    token = bearer(client, endpoint.api_key)
    return ApiEndpointClient(
        client, endpoint, component=component, token=token, policy=policy
    )


def token_text(token: str) -> httpx.Response:
    return httpx.Response(200, text=token)


def passages(count: int) -> list[ScoredPassage]:
    return [
        ScoredPassage(id=f"p{i}", text=f"text {i}", retrieval_score=1.0 - i / 10)
        for i in range(count)
    ]
