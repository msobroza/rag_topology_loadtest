"""``ApiEndpointClient``: auth, the retry policy and the error shape of every call."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import ValidationError

from model_api_fakes import (
    TIMEOUT_S,
    HttpClients,
    ScriptedRoute,
    endpoint_client,
    token_text,
)
from rag_load_test.domain.api_endpoint import DOMINO_RUN_TOKEN, ApiEndpoint
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.standard_formats import CohereRerankResponse
from rag_load_test.infrastructure.endpoint.api_endpoint_client import ApiEndpointClient
from rag_load_test.infrastructure.http.http_policy import HttpPolicy

URL = "http://models.test:8001/v3/rerank"
MODEL = "bge-reranker-base"
PAYLOAD: dict[str, object] = {"model": MODEL, "query": "q", "documents": ["a"]}
OK = httpx.Response(200, json={"results": []})


@pytest.fixture
async def clients() -> AsyncIterator[HttpClients]:
    pool = HttpClients()
    yield pool
    await pool.aclose()


def make(
    clients: HttpClients,
    *script: httpx.Response | Exception,
    api_key: str = "",
    policy: HttpPolicy | None = None,
) -> tuple[ApiEndpointClient, ScriptedRoute]:
    route = ScriptedRoute(*script)
    endpoint = ApiEndpoint(url=URL, api_format="cohere", api_key=api_key, model=MODEL)
    extra = {"policy": policy} if policy else {}
    client = endpoint_client(
        clients.over(route), endpoint, component="reranker", **extra
    )
    return client, route


# --- the request -----------------------------------------------------------


async def test_post_sends_the_payload_with_the_timeout(clients: HttpClients) -> None:
    client, route = make(clients, OK)
    assert await client.post(PAYLOAD) == {"results": []}
    [request] = route.requests
    assert (request.method, request.url, request.json) == ("POST", URL, PAYLOAD)
    assert request.read_timeout == TIMEOUT_S


@pytest.mark.parametrize(
    ("api_key", "authorization"), [("k1", "Bearer k1"), ("", None)]
)
async def test_authorization_header_only_with_a_key(
    clients: HttpClients, api_key: str, authorization: str | None
) -> None:
    client, route = make(clients, OK, api_key=api_key)
    await client.post(PAYLOAD)
    assert route.requests[0].authorization == authorization


async def test_endpoint_and_described(clients: HttpClients) -> None:
    client, _ = make(clients, OK)
    assert client.endpoint == ApiEndpoint(URL, "cohere", "", MODEL)
    assert client.described() == f"{MODEL} at {URL}"


# --- retries ---------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(503, text="busy"),
        httpx.Response(429, text="slow down"),
        httpx.ConnectError("refused"),
    ],
    ids=["5xx", "429", "transport"],
)
async def test_retries_then_succeeds(
    clients: HttpClients, failure: httpx.Response | Exception
) -> None:
    client, route = make(clients, failure, failure, OK)
    assert await client.post(PAYLOAD) == {"results": []}
    assert len(route.requests) == 3


async def test_401_refetches_the_domino_token_then_succeeds(
    clients: HttpClients,
) -> None:
    client, route = make(
        clients,
        token_text("tok-1"),
        httpx.Response(401, text="expired"),
        token_text("tok-2"),
        OK,
        api_key=DOMINO_RUN_TOKEN,
    )
    await client.post(PAYLOAD)
    assert route.methods == ["GET", "POST", "GET", "POST"]
    posts = [r.authorization for r in route.requests if r.method == "POST"]
    assert posts == ["Bearer tok-1", "Bearer tok-2"]


@pytest.mark.parametrize("status", [400, 401, 403])
async def test_a_static_key_rejected_fails_at_once(
    clients: HttpClients, status: int
) -> None:
    client, route = make(clients, httpx.Response(status, text="no"), api_key="bad")
    with pytest.raises(RagDependencyError) as info:
        await client.post(PAYLOAD)
    assert info.value.detail == f"HTTP {status}: no"
    assert (info.value.component, info.value.target) == ("reranker", URL)
    assert info.value.kind == "unavailable"
    assert len(route.requests) == 1


async def test_exhausted_retries_raise_the_last_failure(clients: HttpClients) -> None:
    client, route = make(clients, httpx.Response(429, text="slow down"))
    with pytest.raises(RagDependencyError) as info:
        await client.post(PAYLOAD)
    assert info.value.detail == "HTTP 429: slow down"
    assert info.value.kind == "unavailable"
    assert len(route.requests) == 3  # first attempt + policy.retries


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectError("refused"), "unavailable"),
    ],
    ids=["read-timeout", "connect-error"],
)
async def test_transport_errors_exhaust_retries(
    clients: HttpClients, error: Exception, kind: str
) -> None:
    client, route = make(clients, error)
    with pytest.raises(RagDependencyError) as info:
        await client.post(PAYLOAD)
    assert len(route.requests) == 3  # the first attempt and 2 retries
    assert (info.value.component, info.value.kind) == ("reranker", kind)
    assert info.value.detail == f"{type(error).__name__}: {error}"
    assert info.value.target == URL


async def test_zero_retries_make_one_attempt(clients: HttpClients) -> None:
    policy = HttpPolicy(timeout_s=TIMEOUT_S, retries=0, backoff_s=0.0)
    client, route = make(clients, httpx.Response(503), policy=policy)
    with pytest.raises(RagDependencyError):
        await client.post(PAYLOAD)
    assert len(route.requests) == 1


async def test_retries_wait_the_policy_backoff_doubling_each_time(
    clients: HttpClients,
) -> None:
    policy = HttpPolicy(timeout_s=30.0, retries=2, backoff_s=0.05)
    client, route = make(clients, httpx.Response(503, text="busy"), policy=policy)
    started = time.perf_counter()
    with pytest.raises(RagDependencyError):
        await client.post(PAYLOAD)
    assert time.perf_counter() - started >= 0.15  # 0.05 s, then 0.1 s
    assert route.methods == ["POST"] * 3


@pytest.mark.parametrize(
    "failure",
    [httpx.Response(503, text="busy"), httpx.ReadTimeout("slow")],
    ids=["5xx", "timeout"],
)
async def test_post_once_never_retries_but_keeps_the_timeout(
    clients: HttpClients, failure: httpx.Response | Exception
) -> None:
    client, route = make(clients, failure, OK)
    with pytest.raises(RagDependencyError):
        await client.post_once(PAYLOAD)
    assert len(route.requests) == 1
    assert route.requests[0].read_timeout == TIMEOUT_S


async def test_post_once_returns_the_body(clients: HttpClients) -> None:
    client, route = make(clients, OK)
    assert await client.post_once(PAYLOAD) == {"results": []}
    assert len(route.requests) == 1


# --- malformed answers -----------------------------------------------------


async def test_a_body_that_is_not_json_is_malformed(clients: HttpClients) -> None:
    client, route = make(clients, httpx.Response(200, text="not json"))
    with pytest.raises(RagDependencyError, match="malformed response") as info:
        await client.post(PAYLOAD)
    assert (info.value.component, info.value.target) == ("reranker", URL)
    assert len(route.requests) == 1  # a malformed body is not retried


async def test_malformed_names_what_broke(clients: HttpClients) -> None:
    client, _ = make(clients, OK)
    with pytest.raises(ValidationError) as info:
        CohereRerankResponse.model_validate({"data": []})
    error = client.malformed(info.value)
    assert error.detail == (
        "malformed response: ValidationError for CohereRerankResponse: "
        "results: Field required"
    )
    assert (error.component, error.target, error.kind) == (
        "reranker",
        URL,
        "unavailable",
    )
