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

from query_processing.adapters.key_source.base_key_source import BaseKeySource
from query_processing.domain.endpoints.api_endpoint import ApiEndpoint
from query_processing.domain.retrieval.passages import ScoredPassage
from query_processing.infrastructure.endpoint.api_endpoint_client import (
    ApiEndpointClient,
)
from query_processing.infrastructure.endpoint.bearer_key import BearerKey
from query_processing.infrastructure.http.http_policy import HttpPolicy

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


class FakeKeySource(BaseKeySource):
    """A key source that gives its keys in turn: the next one once told of a refusal.

    It never asks anything, so it is always called on the event loop; it
    records each key it was told was refused.
    """

    def __init__(self, *keys: str) -> None:
        self._keys = list(keys)
        self.refusals: list[str] = []

    def key(self, *, may_ask: bool) -> str | None:
        del may_ask
        return self._keys[0]

    def refused(self, key: str) -> None:
        self.refusals.append(key)
        if key == self._keys[0] and len(self._keys) > 1:
            self._keys.pop(0)

    def described(self) -> str:
        return "its key is the fake source's"


def endpoint_client(
    client: httpx.AsyncClient,
    endpoint: ApiEndpoint,
    *,
    component: str,
    key: str = "",
    policy: HttpPolicy = NO_WAIT,  # retries must not sleep in tests
) -> ApiEndpointClient:
    """The client of ``endpoint``, sending ``key`` from a fake key source, or none."""
    bearer = BearerKey(FakeKeySource(key)) if key else None
    return ApiEndpointClient(
        client, endpoint, component=component, key=bearer, policy=policy
    )


def standard_answer(body: dict[str, Any]) -> dict[str, Any]:
    """The standard format's answer to a request body.

    An embedding request is told apart by its ``input``, a rerank request by its
    ``documents``, anything else is a chat request.
    """
    if "input" in body:
        rows = [
            {"index": i, "embedding": [1.0, 0.0]} for i in range(len(body["input"]))
        ]
        return {"data": rows}
    if "documents" in body:
        scores = range(len(body["documents"]))
        return {"results": [{"index": i, "relevance_score": float(i)} for i in scores]}
    return {"choices": [{"message": {"content": "Twenty-five days."}}]}


def passages(count: int) -> list[ScoredPassage]:
    return [
        ScoredPassage(id=f"p{i}", text=f"text {i}", retrieval_score=1.0 - i / 10)
        for i in range(count)
    ]
