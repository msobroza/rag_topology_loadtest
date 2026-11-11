"""Unit tests for ``RerankApiReranker`` over ``httpx.MockTransport``.

Every request is recorded (method, URL, JSON body, Authorization header, read
timeout) and answered from a script, so the Cohere request shape, auth and the
retry policy are checked without Cohere or a gateway.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from rag_load_test.adapters.hosted_models.rerank_api import RerankApiReranker
from rag_load_test.adapters.shared.http_policy import HttpPolicy
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.passages import ScoredPassage
from rag_load_test.domain.ports import RerankerPort

URL = "https://rerank.test/v2/rerank"
MODEL = "rerank-v3.5"
TIMEOUT_S = 12.5
NO_WAIT = HttpPolicy(timeout_s=TIMEOUT_S, retries=2, backoff_s=0.0)


@dataclass
class RecordedRequest:
    method: str
    url: str
    json: Any
    authorization: str | None
    read_timeout: float | None


class ScriptedRerankRoute:
    """MockTransport handler: replies (or raises) in script order; last entry repeats."""

    def __init__(self, *script: httpx.Response | Exception) -> None:
        self._script = list(script)
        self.requests: list[RecordedRequest] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        timeout = request.extensions.get("timeout", {}).get("read")
        self.requests.append(
            RecordedRequest(
                request.method,
                str(request.url),
                body,
                request.headers.get("Authorization"),
                timeout,
            )
        )
        outcome = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


RerankerFactory = Callable[..., tuple[RerankApiReranker, ScriptedRerankRoute]]


@pytest.fixture
async def make_reranker() -> AsyncIterator[RerankerFactory]:
    """Builds a reranker over a scripted route; closes every client at teardown."""
    clients: list[httpx.AsyncClient] = []

    def make(
        *script: httpx.Response | Exception,
        api_key: str = "",
        policy: HttpPolicy = NO_WAIT,
    ) -> tuple[RerankApiReranker, ScriptedRerankRoute]:
        route = ScriptedRerankRoute(*script)
        client = httpx.AsyncClient(transport=httpx.MockTransport(route))
        clients.append(client)
        reranker = RerankApiReranker(client, URL, MODEL, api_key=api_key, policy=policy)
        return reranker, route

    yield make
    for client in clients:
        await client.aclose()


def rerank_body(*results: tuple[int, float]) -> httpx.Response:
    rows = [{"index": i, "relevance_score": s} for i, s in results]
    # Cohere adds ``id`` and ``meta``; the reader must ignore them.
    return httpx.Response(200, json={"id": "r1", "results": rows, "meta": {}})


def passages(count: int) -> list[ScoredPassage]:
    return [
        ScoredPassage(id=f"p{i}", text=f"text {i}", retrieval_score=1.0 - i / 10)
        for i in range(count)
    ]


# --- request shape and result mapping -------------------------------------


async def test_reranker_is_a_reranker_port(make_reranker: RerankerFactory) -> None:
    reranker, _ = make_reranker(rerank_body())
    assert isinstance(reranker, RerankerPort)


async def test_posts_the_cohere_shape_and_orders_by_relevance(
    make_reranker: RerankerFactory,
) -> None:
    reranker, route = make_reranker(rerank_body((2, 0.9), (0, 0.4)), api_key="k1")
    ranked = await reranker.rerank("vacation", passages(3), top_n=2)
    assert [(p.id, p.rerank_score) for p in ranked] == [("p2", 0.9), ("p0", 0.4)]
    [request] = route.requests
    assert (request.method, request.url) == ("POST", URL)
    assert request.json == {
        "model": MODEL,
        "query": "vacation",
        "documents": ["text 0", "text 1", "text 2"],
        "top_n": 2,
    }
    assert request.read_timeout == TIMEOUT_S


@pytest.mark.parametrize(
    ("api_key", "authorization"), [("k1", "Bearer k1"), ("", None)]
)
async def test_authorization_header_only_with_a_key(
    make_reranker: RerankerFactory, api_key: str, authorization: str | None
) -> None:
    reranker, route = make_reranker(rerank_body((0, 0.5)), api_key=api_key)
    await reranker.rerank("q", passages(1), top_n=1)
    assert route.requests[0].authorization == authorization


async def test_empty_passages_make_no_request(make_reranker: RerankerFactory) -> None:
    reranker, route = make_reranker(rerank_body())
    assert await reranker.rerank("q", [], top_n=3) == []
    assert route.requests == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json={"results": [{"index": "x"}]}),
        httpx.Response(200, text="not json"),
    ],
    ids=["no-results", "bad-row", "not-json"],
)
async def test_malformed_body_raises(
    make_reranker: RerankerFactory, response: httpx.Response
) -> None:
    reranker, route = make_reranker(response)
    with pytest.raises(RagDependencyError, match="malformed response") as info:
        await reranker.rerank("q", passages(2), top_n=2)
    assert (info.value.component, info.value.target) == ("reranker", URL)
    assert len(route.requests) == 1  # a malformed body is not retried


@pytest.mark.parametrize("index", [2, -1])
async def test_out_of_range_index_raises(
    make_reranker: RerankerFactory, index: int
) -> None:
    reranker, _ = make_reranker(rerank_body((index, 0.9)))
    with pytest.raises(RagDependencyError, match=f"index {index} is outside"):
        await reranker.rerank("q", passages(2), top_n=2)


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
    make_reranker: RerankerFactory, failure: httpx.Response | Exception
) -> None:
    reranker, route = make_reranker(failure, failure, rerank_body((0, 0.7)))
    ranked = await reranker.rerank("q", passages(1), top_n=1)
    assert [p.rerank_score for p in ranked] == [0.7]
    assert len(route.requests) == 3


async def test_exhausted_retries_raise_the_last_failure(
    make_reranker: RerankerFactory,
) -> None:
    reranker, route = make_reranker(httpx.Response(429, text="slow down"))
    with pytest.raises(RagDependencyError, match="HTTP 429") as info:
        await reranker.rerank("q", passages(1), top_n=1)
    assert info.value.kind == "unavailable"
    assert len(route.requests) == 3  # first attempt + policy.retries


@pytest.mark.parametrize("status", [400, 401, 403])
async def test_other_4xx_fail_without_retry(
    make_reranker: RerankerFactory, status: int
) -> None:
    reranker, route = make_reranker(httpx.Response(status, text="no"))
    with pytest.raises(RagDependencyError, match=f"HTTP {status}") as info:
        await reranker.rerank("q", passages(1), top_n=1)
    assert (info.value.component, info.value.target) == ("reranker", URL)
    assert len(route.requests) == 1


async def test_timeouts_map_to_kind_timeout(make_reranker: RerankerFactory) -> None:
    reranker, route = make_reranker(httpx.ReadTimeout("slow"))
    with pytest.raises(RagDependencyError) as info:
        await reranker.rerank("q", passages(1), top_n=1)
    assert info.value.kind == "timeout"
    assert len(route.requests) == 3


async def test_zero_retries_make_one_attempt(make_reranker: RerankerFactory) -> None:
    policy = HttpPolicy(timeout_s=TIMEOUT_S, retries=0, backoff_s=0.0)
    reranker, route = make_reranker(httpx.Response(503), policy=policy)
    with pytest.raises(RagDependencyError):
        await reranker.rerank("q", passages(1), top_n=1)
    assert len(route.requests) == 1


# --- readiness -------------------------------------------------------------


async def test_ready_makes_no_request(make_reranker: RerankerFactory) -> None:
    reranker, route = make_reranker(rerank_body())
    assert await reranker.ready() == (True, f"{MODEL} at {URL}")
    assert route.requests == []
