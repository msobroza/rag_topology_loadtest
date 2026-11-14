"""``CompositionRoot``: each backend of each model component opens the right adapter, and
the ``api`` backends call their endpoint with its URL, key, model and policy."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from rag_load_test.adapters.embedder.api.api_embedder import ApiEmbedder
from rag_load_test.adapters.embedder.fake.fake_embedder import FakeEmbedder
from rag_load_test.adapters.llm.api.api_llm import ApiLLM
from rag_load_test.adapters.llm.fake.fake_llm import FakeLLM
from rag_load_test.adapters.reranker.api.api_reranker import ApiReranker
from rag_load_test.adapters.reranker.fake.fake_reranker import FakeReranker
from rag_load_test.adapters.reranker.none.no_reranker import NoReranker
from rag_load_test.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from rag_load_test.composition.composition_root import CompositionRoot
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.passages import ScoredPassage

SettingsFactory = Callable[..., RagSettings]
EMBED_URL = "http://embedder.test/v3/embeddings"
RERANK_URL = "https://rerank.test/v2/rerank"
CHAT_URL = "https://gateway.test/v1/chat/completions"
EVERYTHING_FAKE = {
    "embedding_backend": "fake",
    "rerank_backend": "fake",
    "generation_backend": "fake",
    "vector_store_backend": "fake",
}


class RecordingModelRoutes:
    """Answers the three routes in their formats and records every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        if str(request.url) == EMBED_URL:
            rows = [{"index": i, "embedding": [1.0]} for i in range(len(body["input"]))]
            return httpx.Response(200, json={"data": rows})
        if str(request.url) == RERANK_URL:
            results = [{"index": 0, "relevance_score": 0.9}]
            return httpx.Response(200, json={"results": results})
        answer = {"choices": [{"message": {"content": "Twenty-five days."}}]}
        return httpx.Response(200, json=answer)

    def sent(self) -> list[tuple[str, dict[str, object], str | None, float]]:
        return [
            (
                str(r.url),
                json.loads(r.content),
                r.headers.get("Authorization"),
                r.extensions["timeout"]["read"],
            )
            for r in self.requests
        ]


def candidates() -> list[ScoredPassage]:
    return [
        ScoredPassage(id="a", text="vacation days", retrieval_score=0.5),
        ScoredPassage(id="b", text="expense receipts", retrieval_score=0.4),
    ]


# --- each backend of each component ------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "adapter"),
    [("fake", FakeEmbedder), ("api", ApiEmbedder)],
)
async def test_each_embedding_backend_opens_its_adapter(
    settings_factory: SettingsFactory, backend: str, adapter: type
) -> None:
    settings = settings_factory(
        **(EVERYTHING_FAKE | {"embedding_backend": backend}),
        embedding_api_url=EMBED_URL,
    )
    async with CompositionRoot(
        settings, vector_store=InMemoryVectorStore()
    ) as composition:
        assert isinstance((await composition.workflow_dependencies()).embedder, adapter)


@pytest.mark.parametrize(
    ("backend", "adapter"),
    [("fake", FakeReranker), ("none", NoReranker), ("api", ApiReranker)],
)
async def test_each_rerank_backend_opens_its_adapter(
    settings_factory: SettingsFactory, backend: str, adapter: type
) -> None:
    settings = settings_factory(
        **(EVERYTHING_FAKE | {"rerank_backend": backend}), rerank_api_url=RERANK_URL
    )
    async with CompositionRoot(settings) as composition:
        assert isinstance((await composition.workflow_dependencies()).reranker, adapter)


@pytest.mark.parametrize(("backend", "adapter"), [("fake", FakeLLM), ("api", ApiLLM)])
async def test_each_generation_backend_opens_its_adapter(
    settings_factory: SettingsFactory, backend: str, adapter: type
) -> None:
    settings = settings_factory(**(EVERYTHING_FAKE | {"generation_backend": backend}))
    async with CompositionRoot(settings) as composition:
        assert isinstance((await composition.workflow_dependencies()).llm, adapter)


# --- the api backends on the wire --------------------------------------------------


async def test_the_api_embedder_calls_its_endpoint(
    settings_factory: SettingsFactory,
) -> None:
    routes = RecordingModelRoutes()
    settings = settings_factory(
        **(EVERYTHING_FAKE | {"embedding_backend": "api"}),
        embedding_api_url=EMBED_URL,
        embedding_api_key="emb-key",
        embedding_api_model="bge-small",
        embedding_batch_size=2,
        http_timeout_s=4.5,
    )
    transport = httpx.MockTransport(routes)
    composition = CompositionRoot(
        settings, vector_store=InMemoryVectorStore(), transport=transport
    )
    async with composition:
        embedder, _ = await composition.ingest_ports()
        assert await embedder.embed(["a", "b", "c"]) == [[1.0], [1.0], [1.0]]

    assert routes.sent() == [
        (EMBED_URL, {"model": "bge-small", "input": ["a", "b"]}, "Bearer emb-key", 4.5),
        (EMBED_URL, {"model": "bge-small", "input": ["c"]}, "Bearer emb-key", 4.5),
    ]


async def test_the_api_reranker_calls_its_endpoint(
    settings_factory: SettingsFactory,
) -> None:
    routes = RecordingModelRoutes()
    settings = settings_factory(
        **(EVERYTHING_FAKE | {"rerank_backend": "api"}),
        rerank_api_url=RERANK_URL,
        rerank_api_key="co-key",
        rerank_api_model="rerank-v3.5",
        http_timeout_s=4.5,
    )
    async with CompositionRoot(
        settings, transport=httpx.MockTransport(routes)
    ) as composition:
        reranker = (await composition.workflow_dependencies()).reranker
        [best] = await reranker.rerank("vacation", candidates(), top_n=1)

    assert (best.id, best.rerank_score) == ("a", 0.9)
    expected_body = {
        "model": "rerank-v3.5",
        "query": "vacation",
        "documents": ["vacation days", "expense receipts"],
        "top_n": 1,
    }
    assert routes.sent() == [(RERANK_URL, expected_body, "Bearer co-key", 4.5)]


async def test_the_api_llm_calls_its_endpoint_with_the_generation_timeout(
    settings_factory: SettingsFactory,
) -> None:
    routes = RecordingModelRoutes()
    settings = settings_factory(
        **(EVERYTHING_FAKE | {"generation_backend": "api"}),
        generation_api_url=CHAT_URL,
        generation_api_key="sk-key",
        generation_api_model="llama-3",
        generation_temperature=0.7,
        generation_timeout_s=90.0,
        http_timeout_s=4.5,
    )
    async with CompositionRoot(
        settings, transport=httpx.MockTransport(routes)
    ) as composition:
        llm = (await composition.workflow_dependencies()).llm
        messages = [{"role": "user", "content": "How many vacation days?"}]
        assert await llm.generate(messages) == "Twenty-five days."

    expected_body = {"model": "llama-3", "messages": messages, "temperature": 0.7}
    assert routes.sent() == [(CHAT_URL, expected_body, "Bearer sk-key", 90.0)]


async def test_the_generation_endpoint_keeps_the_shared_retries(
    settings_factory: SettingsFactory,
) -> None:
    attempts: list[str] = []

    def busy(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        return httpx.Response(503, text="busy")

    settings = settings_factory(
        **(EVERYTHING_FAKE | {"generation_backend": "api"}),
        http_retries=1,
        http_backoff_s=0.0,
    )
    async with CompositionRoot(
        settings, transport=httpx.MockTransport(busy)
    ) as composition:
        llm = (await composition.workflow_dependencies()).llm
        with pytest.raises(RagDependencyError, match="HTTP 503"):
            await llm.generate([{"role": "user", "content": "q"}])

    assert len(attempts) == 2  # the first attempt and http_retries=1
