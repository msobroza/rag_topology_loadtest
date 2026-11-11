"""``BearerToken``: no header, a static key, or Domino's run token, cached for a TTL."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from model_api_fakes import (
    TIMEOUT_S,
    TOKEN_URL,
    HttpClients,
    ScriptedRoute,
    bearer,
    token_text,
)
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.model_api import DOMINO_RUN_TOKEN


@pytest.fixture
async def clients() -> AsyncIterator[HttpClients]:
    pool = HttpClients()
    yield pool
    await pool.aclose()


@pytest.mark.parametrize(
    ("api_key", "expected"), [("", {}), ("abc", {"Authorization": "Bearer abc"})]
)
async def test_static_values_need_no_request(
    clients: HttpClients, api_key: str, expected: dict[str, str]
) -> None:
    route = ScriptedRoute(token_text("unused"))
    assert await bearer(clients.over(route), api_key).headers() == expected
    assert route.requests == []


async def test_domino_fetches_once_and_caches(clients: HttpClients) -> None:
    route = ScriptedRoute(token_text("tok-1\n"))
    token = bearer(clients.over(route), DOMINO_RUN_TOKEN)
    first, second = await token.headers(), await token.headers()
    assert first == second == {"Authorization": "Bearer tok-1"}
    assert route.methods == ["GET"]
    assert route.requests[0].url == TOKEN_URL
    assert route.requests[0].authorization is None
    assert route.requests[0].read_timeout == TIMEOUT_S


async def test_refetches_once_the_ttl_has_passed(clients: HttpClients) -> None:
    route = ScriptedRoute(token_text("tok-1"), token_text("tok-2"))
    token = bearer(clients.over(route), DOMINO_RUN_TOKEN, ttl_s=0.0)  # always stale
    assert await token.headers() == {"Authorization": "Bearer tok-1"}
    assert await token.headers() == {"Authorization": "Bearer tok-2"}


async def test_invalidate_forces_refetch(clients: HttpClients) -> None:
    route = ScriptedRoute(token_text("tok-1"), token_text("tok-2"))
    token = bearer(clients.over(route), DOMINO_RUN_TOKEN)
    assert await token.headers() == {"Authorization": "Bearer tok-1"}
    token.invalidate()
    assert await token.headers() == {"Authorization": "Bearer tok-2"}
    assert route.methods == ["GET", "GET"]


@pytest.mark.parametrize(
    "outcome",
    [
        httpx.Response(500, text="nope"),
        token_text("   "),
        httpx.ConnectError("refused"),
    ],
    ids=["non-200", "empty-body", "transport-error"],
)
async def test_domino_failures_raise(
    clients: HttpClients, outcome: httpx.Response | Exception
) -> None:
    token = bearer(clients.over(ScriptedRoute(outcome)), DOMINO_RUN_TOKEN)
    with pytest.raises(RagDependencyError) as info:
        await token.headers()
    assert info.value.component == "domino_access_token"
    assert info.value.target == TOKEN_URL


@pytest.mark.parametrize(
    ("api_key", "statuses"),
    [(DOMINO_RUN_TOKEN, frozenset({401})), ("abc", frozenset()), ("", frozenset())],
)
async def test_only_the_domino_token_makes_401_retryable(
    clients: HttpClients, api_key: str, statuses: frozenset[int]
) -> None:
    token = bearer(clients.over(ScriptedRoute(token_text("unused"))), api_key)
    assert token.retry_statuses == statuses
