"""``ApiEmbedder`` in the OpenAI embeddings format over ``httpx.MockTransport``."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from model_api_fakes import (
    TIMEOUT_S,
    HttpClients,
    ScriptedRoute,
    endpoint_client,
    token_text,
)
from rag_load_test.adapters.model_api.api_embedder import ApiEmbedder
from rag_load_test.adapters.model_api.openai_embeddings_format import (
    OpenAIEmbeddingsFormat,
)
from rag_load_test.domain.api_endpoint import DOMINO_RUN_TOKEN, ApiEndpoint
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.ports import BaseEmbedder

URL = "http://models.test:8002/v3/embeddings"
MODEL = "bge-small-en-v1.5"


@pytest.fixture
async def clients() -> AsyncIterator[HttpClients]:
    pool = HttpClients()
    yield pool
    await pool.aclose()


def make(
    clients: HttpClients,
    *script: httpx.Response | Exception,
    api_key: str = "",
    batch_size: int = 64,
) -> tuple[ApiEmbedder, ScriptedRoute]:
    route = ScriptedRoute(*script)
    endpoint = ApiEndpoint(url=URL, api_format="openai", api_key=api_key, model=MODEL)
    client = endpoint_client(clients.over(route), endpoint, component="embedder")
    embedder = ApiEmbedder(client, OpenAIEmbeddingsFormat(), batch_size=batch_size)
    return embedder, route


def embeddings_body(vectors: list[list[float]]) -> httpx.Response:
    """OpenAI-style body with rows deliberately in reverse ``index`` order."""
    rows = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    return httpx.Response(200, json={"object": "list", "data": rows[::-1]})


async def test_posts_the_openai_shape_and_reorders_rows(clients: HttpClients) -> None:
    embedder, route = make(clients, embeddings_body([[0.1, 0.2], [0.3, 0.4]]))
    assert isinstance(embedder, BaseEmbedder)
    assert await embedder.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    [request] = route.requests
    assert (request.method, request.url) == ("POST", URL)
    assert request.json == {"model": MODEL, "input": ["a", "b"]}
    assert embedder.model_name == MODEL


async def test_one_text_is_still_sent_as_a_list(clients: HttpClients) -> None:
    embedder, route = make(clients, embeddings_body([[1.0]]))
    await embedder.embed(["only"])
    assert route.requests[0].json["input"] == ["only"]


async def test_batches_requests(clients: HttpClients) -> None:
    embedder, route = make(
        clients,
        embeddings_body([[1.0], [2.0]]),
        embeddings_body([[3.0], [4.0]]),
        embeddings_body([[5.0]]),
        batch_size=2,
    )
    vectors = await embedder.embed([f"t{i}" for i in range(5)])
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert [r.json["input"] for r in route.requests] == [
        ["t0", "t1"],
        ["t2", "t3"],
        ["t4"],
    ]


async def test_no_texts_send_no_request(clients: HttpClients) -> None:
    embedder, route = make(clients, embeddings_body([]))
    assert await embedder.embed([]) == []
    assert route.requests == []


async def test_retries_a_5xx_then_succeeds(clients: HttpClients) -> None:
    embedder, route = make(
        clients, httpx.Response(503, text="busy"), embeddings_body([[1.0]])
    )
    assert await embedder.embed(["a"]) == [[1.0]]
    assert route.methods == ["POST", "POST"]


async def test_domino_key_sends_the_run_token(clients: HttpClients) -> None:
    embedder, route = make(
        clients, token_text("tok-1"), embeddings_body([[1.0]]), api_key=DOMINO_RUN_TOKEN
    )
    await embedder.embed(["a"])
    assert route.methods == ["GET", "POST"]
    assert route.requests[1].authorization == "Bearer tok-1"


async def test_malformed_body_raises(clients: HttpClients) -> None:
    embedder, _ = make(clients, httpx.Response(200, json={"object": "list"}))
    with pytest.raises(RagDependencyError, match="malformed response") as info:
        await embedder.embed(["a"])
    assert info.value.component == "embedder"
    assert info.value.detail == (
        "malformed response: ValidationError for OpenAIEmbeddingsResponse: "
        "data: Field required"
    )
    assert info.value.target == URL


@pytest.mark.parametrize(
    "rows",
    [
        [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}],
        [{"index": 1, "embedding": [1.0]}],
    ],
    ids=["repeated-index", "missing-index"],
)
async def test_row_indexes_must_be_0_to_n_once_each(
    clients: HttpClients, rows: list[dict[str, object]]
) -> None:
    embedder, _ = make(clients, httpx.Response(200, json={"data": rows}))
    with pytest.raises(RagDependencyError, match="row indexes"):
        await embedder.embed(["a", "b"][: len(rows)])


async def test_one_vector_per_text_sent(clients: HttpClients) -> None:
    embedder, _ = make(clients, embeddings_body([[1.0]]))
    with pytest.raises(RagDependencyError, match="1 vectors for 2 texts sent"):
        await embedder.embed(["a", "b"])


# --- readiness -------------------------------------------------------------


async def test_ready_sends_one_real_request(clients: HttpClients) -> None:
    embedder, route = make(clients, embeddings_body([[1.0]]))
    assert await embedder.ready() == (True, f"{MODEL} at {URL}")
    [request] = route.requests
    assert request.json == {"model": MODEL, "input": ["ready"]}
    assert request.read_timeout == TIMEOUT_S


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (httpx.Response(503, text="busy"), "HTTP 503: busy"),
        (httpx.ConnectError("refused"), "ConnectError: refused"),
        (httpx.Response(200, json={"data": "x"}), "malformed response"),
        (httpx.Response(200, json={"data": []}), "0 vectors for 1 texts"),
    ],
    ids=["5xx", "transport", "malformed", "no-vector"],
)
async def test_ready_reports_failure_without_retry_or_raising(
    clients: HttpClients, answer: httpx.Response | Exception, expected: str
) -> None:
    embedder, route = make(clients, answer, embeddings_body([[1.0]]))
    ok, detail = await embedder.ready()
    assert not ok
    assert expected in detail
    assert len(route.requests) == 1


async def test_ready_reports_a_failed_token_fetch(clients: HttpClients) -> None:
    embedder, route = make(
        clients, httpx.Response(500, text="nope"), api_key=DOMINO_RUN_TOKEN
    )
    ok, detail = await embedder.ready()
    assert not ok
    assert "domino_access_token" in detail
    assert route.methods == ["GET"]
