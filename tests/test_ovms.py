"""Unit tests for ``rag_load_test.adapters.ovms.ovms_adapters`` over ``httpx.MockTransport``.

Every request is recorded (method, URL, JSON body, Authorization header) and
answered from a script, so auth, batching, ordering and the retry policy are
checked without a model server or a Domino token endpoint.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from rag_load_test.adapters.ovms.ovms_adapters import (
    BearerToken,
    OvmsEmbedder,
    OvmsEndpoint,
    OvmsReranker,
)
from rag_load_test.adapters.shared.http_policy import HttpPolicy
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.passages import ScoredPassage

BASE_URL = "http://ovms.test:8001"
RERANK_MODEL = "bge-reranker-base"
EMBED_MODEL = "bge-small-en-v1.5"
TOKEN_URL = "http://token.test/access-token"
RETRIES, TIMEOUT_S = 2, 30.0
NO_WAIT = HttpPolicy(timeout_s=TIMEOUT_S, retries=RETRIES, backoff_s=0.0)


@dataclass
class RecordedRequest:
    method: str
    url: str
    json: Any
    authorization: str | None
    read_timeout: float | None


class ScriptedTransport:
    """MockTransport handler: replies (or raises) in script order; last entry repeats."""

    def __init__(self, *script: httpx.Response | Exception) -> None:
        self._script = list(script)
        self.requests: list[RecordedRequest] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(_record(request))
        outcome = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def methods(self) -> list[str]:
        return [r.method for r in self.requests]


def _record(request: httpx.Request) -> RecordedRequest:
    body = json.loads(request.content) if request.content else None
    authorization = request.headers.get("Authorization")
    read_timeout = request.extensions.get("timeout", {}).get("read")
    return RecordedRequest(
        request.method, str(request.url), body, authorization, read_timeout
    )


ClientFactory = Callable[[ScriptedTransport], httpx.AsyncClient]


@pytest.fixture
async def client_factory() -> AsyncIterator[ClientFactory]:
    """Builds clients over a scripted transport; closes them all at teardown."""
    clients: list[httpx.AsyncClient] = []

    def make(script: ScriptedTransport) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(script))
        clients.append(client)
        return client

    yield make
    for client in clients:
        await client.aclose()


def bearer(
    client: httpx.AsyncClient, auth: str, *, ttl_s: float = 240.0
) -> BearerToken:
    return BearerToken(client, auth, token_url=TOKEN_URL, ttl_s=ttl_s, policy=NO_WAIT)


def endpoint(
    client: httpx.AsyncClient,
    kind: type[OvmsEndpoint] = OvmsReranker,
    *,
    auth: str = "",
    model: str = RERANK_MODEL,
    base_url: str = BASE_URL,
    policy: HttpPolicy = NO_WAIT,  # retries must not sleep in tests
    batch_size: int = 64,
) -> Any:
    sizing = {"batch_size": batch_size} if kind is OvmsEmbedder else {}
    token = bearer(client, auth)
    return kind(client, base_url, model, token=token, policy=policy, **sizing)


def token_text(token: str) -> httpx.Response:
    return httpx.Response(200, text=token)


def embeddings_body(vectors: list[list[float]]) -> httpx.Response:
    """OpenAI-style body with rows deliberately in reverse ``index`` order."""
    rows = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    return httpx.Response(200, json={"object": "list", "data": rows[::-1]})


def rerank_body(*results: tuple[int, float]) -> httpx.Response:
    rows = [{"index": i, "relevance_score": s} for i, s in results]
    return httpx.Response(200, json={"results": rows})


def passages(count: int) -> list[ScoredPassage]:
    return [
        ScoredPassage(id=f"p{i}", text=f"text {i}", retrieval_score=1.0 - i / 10)
        for i in range(count)
    ]


# --- BearerToken -----------------------------------------------------------


@pytest.mark.parametrize(
    ("auth", "expected"), [("", {}), ("abc", {"Authorization": "Bearer abc"})]
)
async def test_bearer_token_static_values_need_no_request(
    client_factory: ClientFactory, auth: str, expected: dict[str, str]
) -> None:
    script = ScriptedTransport(token_text("unused"))
    assert await bearer(client_factory(script), auth).headers() == expected
    assert script.requests == []


async def test_bearer_token_domino_fetches_once_and_caches(
    client_factory: ClientFactory,
) -> None:
    script = ScriptedTransport(token_text("tok-1\n"))
    token = bearer(client_factory(script), "domino")
    first, second = await token.headers(), await token.headers()
    assert first == second == {"Authorization": "Bearer tok-1"}
    assert script.methods == ["GET"]
    assert script.requests[0].url == TOKEN_URL
    assert script.requests[0].authorization is None
    assert script.requests[0].read_timeout == TIMEOUT_S


async def test_bearer_token_refetches_once_the_ttl_has_passed(
    client_factory: ClientFactory,
) -> None:
    script = ScriptedTransport(token_text("tok-1"), token_text("tok-2"))
    token = bearer(client_factory(script), "domino", ttl_s=0.0)  # always stale
    assert await token.headers() == {"Authorization": "Bearer tok-1"}
    assert await token.headers() == {"Authorization": "Bearer tok-2"}


async def test_bearer_token_invalidate_forces_refetch(
    client_factory: ClientFactory,
) -> None:
    script = ScriptedTransport(token_text("tok-1"), token_text("tok-2"))
    token = bearer(client_factory(script), "domino")
    assert await token.headers() == {"Authorization": "Bearer tok-1"}
    token.invalidate()
    assert await token.headers() == {"Authorization": "Bearer tok-2"}
    assert script.methods == ["GET", "GET"]


@pytest.mark.parametrize(
    "outcome",
    [
        httpx.Response(500, text="nope"),
        token_text("   "),
        httpx.ConnectError("refused"),
    ],
    ids=["non-200", "empty-body", "transport-error"],
)
async def test_bearer_token_domino_failures_raise(
    client_factory: ClientFactory, outcome: httpx.Response | Exception
) -> None:
    token = bearer(client_factory(ScriptedTransport(outcome)), "domino")
    with pytest.raises(RagDependencyError) as info:
        await token.headers()
    assert info.value.component == "domino_access_token"
    assert info.value.target == TOKEN_URL


# --- OvmsEmbedder ----------------------------------------------------------


async def test_embedder_posts_payload_reorders_rows_and_strips_trailing_slash(
    client_factory: ClientFactory,
) -> None:
    script = ScriptedTransport(embeddings_body([[0.1, 0.2], [0.3, 0.4]]))
    embedder = endpoint(
        client_factory(script),
        OvmsEmbedder,
        auth="abc",
        model=EMBED_MODEL,
        base_url=BASE_URL + "/",
    )
    assert await embedder.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    [request] = script.requests
    assert (request.method, request.url) == ("POST", f"{BASE_URL}/v3/embeddings")
    assert request.json == {"model": EMBED_MODEL, "input": ["a", "b"]}
    assert request.authorization == "Bearer abc"
    assert embedder.model_name == f"ovms:{EMBED_MODEL}"


async def test_embedder_batches_requests(client_factory: ClientFactory) -> None:
    script = ScriptedTransport(
        embeddings_body([[1.0], [2.0]]),
        embeddings_body([[3.0], [4.0]]),
        embeddings_body([[5.0]]),
    )
    embedder = endpoint(
        client_factory(script), OvmsEmbedder, model=EMBED_MODEL, batch_size=2
    )
    vectors = await embedder.embed([f"t{i}" for i in range(5)])
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert script.methods == ["POST"] * 3
    assert [r.json["input"] for r in script.requests] == [
        ["t0", "t1"],
        ["t2", "t3"],
        ["t4"],
    ]


async def test_embedder_malformed_body_raises(client_factory: ClientFactory) -> None:
    script = ScriptedTransport(httpx.Response(200, json={"object": "list"}))
    embedder = endpoint(client_factory(script), OvmsEmbedder, model=EMBED_MODEL)
    with pytest.raises(RagDependencyError) as info:
        await embedder.embed(["a"])
    assert info.value.component == "embedder"
    # The shape check now lives in v3_contract (pydantic), not in a KeyError.
    assert info.value.detail == (
        "malformed response: ValidationError for EmbeddingsResponse: "
        "data: Field required"
    )
    assert info.value.target == f"{BASE_URL}/v3/embeddings"


# --- OvmsReranker ----------------------------------------------------------


async def test_reranker_posts_payload_and_orders_by_relevance(
    client_factory: ClientFactory,
) -> None:
    script = ScriptedTransport(rerank_body((0, 0.5), (2, 0.9)))
    reranker = endpoint(client_factory(script))
    ranked = await reranker.rerank("q", passages(3), top_n=2)
    assert [p.id for p in ranked] == ["p2", "p0"]
    assert [p.rerank_score for p in ranked] == [0.9, 0.5]
    assert [p.retrieval_score for p in ranked] == [0.8, 1.0]
    [request] = script.requests
    assert request.url == f"{BASE_URL}/v3/rerank"
    assert request.json == {
        "model": RERANK_MODEL,
        "query": "q",
        "documents": ["text 0", "text 1", "text 2"],
        "top_n": 2,
    }


async def test_reranker_empty_passages_make_no_request(
    client_factory: ClientFactory,
) -> None:
    script = ScriptedTransport(rerank_body())
    assert await endpoint(client_factory(script)).rerank("q", [], top_n=2) == []
    assert script.requests == []


async def test_reranker_malformed_body_raises(client_factory: ClientFactory) -> None:
    script = ScriptedTransport(httpx.Response(200, json={"data": []}))
    with pytest.raises(RagDependencyError) as info:
        await endpoint(client_factory(script)).rerank("q", passages(2), top_n=1)
    assert info.value.component == "reranker"
    assert info.value.detail == (
        "malformed response: ValidationError for RerankResponse: "
        "results: Field required"
    )
    assert info.value.target == f"{BASE_URL}/v3/rerank"


async def test_reranker_out_of_range_index_raises(
    client_factory: ClientFactory,
) -> None:
    script = ScriptedTransport(rerank_body((7, 0.9)))
    with pytest.raises(RagDependencyError) as info:
        await endpoint(client_factory(script)).rerank("q", passages(2), top_n=1)
    assert info.value.component == "reranker"
    assert info.value.detail.startswith("malformed response: IndexError")


# --- retry policy (_post) --------------------------------------------------


async def test_post_retries_5xx_then_succeeds(client_factory: ClientFactory) -> None:
    script = ScriptedTransport(
        httpx.Response(503, text="busy"), rerank_body((0, 0.9), (1, 0.1))
    )
    ranked = await endpoint(client_factory(script)).rerank("q", passages(2), top_n=2)
    assert [p.id for p in ranked] == ["p0", "p1"]
    assert script.methods == ["POST", "POST"]


async def test_post_401_refetches_domino_token_then_succeeds(
    client_factory: ClientFactory,
) -> None:
    script = ScriptedTransport(
        token_text("tok-1"),
        httpx.Response(401, text="expired"),
        token_text("tok-2"),
        rerank_body((0, 0.9)),
    )
    reranker = endpoint(client_factory(script), auth="domino")
    ranked = await reranker.rerank("q", passages(1), top_n=1)
    assert [p.id for p in ranked] == ["p0"]
    assert script.methods == ["GET", "POST", "GET", "POST"]
    posts = [r.authorization for r in script.requests if r.method == "POST"]
    assert posts == ["Bearer tok-1", "Bearer tok-2"]


async def test_post_400_fails_without_retry(client_factory: ClientFactory) -> None:
    script = ScriptedTransport(httpx.Response(400, text="bad request"))
    with pytest.raises(RagDependencyError) as info:
        await endpoint(client_factory(script)).rerank("q", passages(1), top_n=1)
    assert script.methods == ["POST"]
    assert info.value.detail == "HTTP 400: bad request"
    assert info.value.kind == "unavailable"
    assert info.value.target == f"{BASE_URL}/v3/rerank"


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectError("refused"), "unavailable"),
    ],
    ids=["read-timeout", "connect-error"],
)
async def test_post_transport_errors_exhaust_retries(
    client_factory: ClientFactory, error: Exception, kind: str
) -> None:
    script = ScriptedTransport(error)
    with pytest.raises(RagDependencyError) as info:
        await endpoint(client_factory(script)).rerank("q", passages(1), top_n=1)
    assert len(script.requests) == 3  # the first attempt and RETRIES=2 retries
    assert info.value.component == "reranker"
    assert info.value.kind == kind
    assert info.value.detail == f"{type(error).__name__}: {error}"
    assert info.value.target == f"{BASE_URL}/v3/rerank"


# --- ready() ---------------------------------------------------------------


async def test_ready_reports_200(client_factory: ClientFactory) -> None:
    script = ScriptedTransport(httpx.Response(200, json={"name": RERANK_MODEL}))
    ok, detail = await endpoint(client_factory(script)).ready()
    model_url = f"{BASE_URL}/v3/models/{RERANK_MODEL}"
    assert (ok, detail) == (True, f"{model_url} -> 200")
    assert script.requests[0].method == "GET"
    assert script.requests[0].url == model_url


async def test_ready_reports_connect_error(client_factory: ClientFactory) -> None:
    script = ScriptedTransport(httpx.ConnectError("refused"))
    ok, detail = await endpoint(client_factory(script)).ready()
    assert (ok, detail) == (False, "ConnectError: refused")


async def test_post_with_zero_retries_makes_one_attempt(
    client_factory: ClientFactory,
) -> None:
    script = ScriptedTransport(httpx.Response(503, text="busy"))
    reranker = endpoint(client_factory(script), policy=HttpPolicy(4.5, 0, 0.0))
    with pytest.raises(RagDependencyError):
        await reranker.rerank("q", passages(1), top_n=1)
    assert script.methods == ["POST"]


async def test_calls_carry_the_configured_timeout(
    client_factory: ClientFactory,
) -> None:
    script = ScriptedTransport(rerank_body((0, 0.9)))
    reranker = endpoint(client_factory(script), policy=HttpPolicy(4.5, 2, 0.0))
    await reranker.rerank("q", passages(1), top_n=1)
    assert await reranker.ready() == (
        True,
        f"{BASE_URL}/v3/models/{RERANK_MODEL} -> 200",
    )
    assert [r.read_timeout for r in script.requests] == [4.5, 4.5]


async def test_retries_wait_the_policy_backoff_doubling_each_time(
    client_factory: ClientFactory,
) -> None:
    script = ScriptedTransport(httpx.Response(503, text="busy"))
    reranker = endpoint(client_factory(script), policy=HttpPolicy(30.0, 2, 0.05))
    started = time.perf_counter()
    with pytest.raises(RagDependencyError):
        await reranker.rerank("q", passages(1), top_n=1)
    assert time.perf_counter() - started >= 0.15  # 0.05 s, then 0.1 s
    assert script.methods == ["POST"] * 3
