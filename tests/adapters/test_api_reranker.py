"""``ApiReranker`` in the Cohere rerank format over ``httpx.MockTransport``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from model_api_fakes import (
    TIMEOUT_S,
    HttpClients,
    ScriptedRoute,
    endpoint_client,
    passages,
)
from rag_load_test.adapters.model_api.api_reranker import ApiReranker
from rag_load_test.adapters.model_api.cohere_rerank_format import CohereRerankFormat
from rag_load_test.domain.api_endpoint import ApiEndpoint
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.ports import BaseReranker

URL = "https://rerank.test/v2/rerank"
MODEL = "rerank-v3.5"


@pytest.fixture
async def clients() -> AsyncIterator[HttpClients]:
    pool = HttpClients()
    yield pool
    await pool.aclose()


def make(
    clients: HttpClients, *script: httpx.Response | Exception, api_key: str = ""
) -> tuple[ApiReranker, ScriptedRoute]:
    route = ScriptedRoute(*script)
    endpoint = ApiEndpoint(url=URL, api_format="cohere", api_key=api_key, model=MODEL)
    client = endpoint_client(clients.over(route), endpoint, component="reranker")
    return ApiReranker(client, CohereRerankFormat()), route


def rerank_body(*results: tuple[int, float]) -> httpx.Response:
    rows = [{"index": i, "relevance_score": s} for i, s in results]
    # Cohere adds ``id`` and ``meta``; the reader must ignore them.
    return httpx.Response(200, json={"id": "r1", "results": rows, "meta": {}})


async def test_posts_the_cohere_shape_and_orders_by_relevance(
    clients: HttpClients,
) -> None:
    reranker, route = make(clients, rerank_body((2, 0.9), (0, 0.4)), api_key="k1")
    assert isinstance(reranker, BaseReranker)
    ranked = await reranker.rerank("vacation", passages(3), top_n=2)
    assert [(p.id, p.rerank_score) for p in ranked] == [("p2", 0.9), ("p0", 0.4)]
    assert [p.retrieval_score for p in ranked] == [0.8, 1.0]
    [request] = route.requests
    assert (request.method, request.url) == ("POST", URL)
    assert request.json == {
        "model": MODEL,
        "query": "vacation",
        "documents": ["text 0", "text 1", "text 2"],
        "top_n": 2,
    }
    assert request.authorization == "Bearer k1"


async def test_keeps_top_n_even_if_the_server_sends_more(clients: HttpClients) -> None:
    reranker, _ = make(clients, rerank_body((0, 0.1), (1, 0.7), (2, 0.5)))
    ranked = await reranker.rerank("q", passages(3), top_n=1)
    assert [p.id for p in ranked] == ["p1"]


async def test_empty_passages_make_no_request(clients: HttpClients) -> None:
    reranker, route = make(clients, rerank_body())
    assert await reranker.rerank("q", [], top_n=3) == []
    assert route.requests == []


async def test_retries_a_429_then_succeeds(clients: HttpClients) -> None:
    reranker, route = make(
        clients, httpx.Response(429, text="slow down"), rerank_body((0, 0.7))
    )
    ranked = await reranker.rerank("q", passages(1), top_n=1)
    assert [p.rerank_score for p in ranked] == [0.7]
    assert len(route.requests) == 2


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
    clients: HttpClients, response: httpx.Response
) -> None:
    reranker, route = make(clients, response)
    with pytest.raises(RagDependencyError, match="malformed response") as info:
        await reranker.rerank("q", passages(2), top_n=2)
    assert (info.value.component, info.value.target) == ("reranker", URL)
    assert len(route.requests) == 1  # a malformed body is not retried


@pytest.mark.parametrize("index", [2, -1])
async def test_out_of_range_index_raises(clients: HttpClients, index: int) -> None:
    reranker, _ = make(clients, rerank_body((index, 0.9)))
    with pytest.raises(RagDependencyError, match=f"index {index} is outside") as info:
        await reranker.rerank("q", passages(2), top_n=2)
    assert info.value.detail.startswith("malformed response: IndexError")


async def test_timeouts_map_to_kind_timeout(clients: HttpClients) -> None:
    reranker, route = make(clients, httpx.ReadTimeout("slow"))
    with pytest.raises(RagDependencyError) as info:
        await reranker.rerank("q", passages(1), top_n=1)
    assert (info.value.component, info.value.kind) == ("reranker", "timeout")
    assert len(route.requests) == 3


# --- readiness -------------------------------------------------------------


async def test_ready_sends_one_real_request(clients: HttpClients) -> None:
    reranker, route = make(clients, rerank_body((0, 0.5)))
    assert await reranker.ready() == (True, f"{MODEL} at {URL}")
    [request] = route.requests
    assert request.json == {
        "model": MODEL,
        "query": "ready",
        "documents": ["ready"],
        "top_n": 1,
    }
    assert request.read_timeout == TIMEOUT_S


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (httpx.Response(503, text="busy"), "HTTP 503: busy"),
        (httpx.Response(401, text="bad key"), "HTTP 401: bad key"),
        (httpx.ReadTimeout("slow"), "reranker timeout"),
        (httpx.Response(200, json={"data": []}), "malformed response"),
        (rerank_body((3, 0.5)), "index 3 is outside"),
    ],
    ids=["5xx", "rejected-key", "timeout", "malformed", "bad-index"],
)
async def test_ready_reports_failure_without_retry_or_raising(
    clients: HttpClients, answer: httpx.Response | Exception, expected: str
) -> None:
    reranker, route = make(clients, answer, rerank_body((0, 0.5)), api_key="k1")
    ok, detail = await reranker.ready()
    assert not ok
    assert expected in detail
    assert len(route.requests) == 1
