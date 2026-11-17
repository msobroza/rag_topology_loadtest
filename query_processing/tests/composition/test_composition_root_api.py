"""``CompositionRoot``: each backend of each model component opens the right adapter, and
the ``api`` backends call their endpoint with its URL, key, model and policy."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from query_processing.adapters.embedder.api.api_embedder import ApiEmbedder
from query_processing.adapters.embedder.fake.fake_embedder import FakeEmbedder
from query_processing.adapters.llm.api.api_llm import ApiLLM
from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.adapters.reranker.api.api_reranker import ApiReranker
from query_processing.adapters.reranker.fake.fake_reranker import FakeReranker
from query_processing.adapters.reranker.none.no_reranker import NoReranker
from query_processing.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.errors import RagDependencyError
from query_processing.domain.passages import ScoredPassage
from rag_components import FAKE_BACKENDS

SettingsFactory = Callable[..., WorkflowSettings]
EMBED_URL = "http://embedder.test/v3/embeddings"
RERANK_URL = "https://rerank.test/v2/rerank"
CHAT_URL = "https://gateway.test/v1/chat/completions"


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
        **(FAKE_BACKENDS | {"components__embedder__backend": backend}),
        components__embedder__api__url=EMBED_URL,
    )
    async with CompositionRoot(
        settings, vector_store=InMemoryVectorStore()
    ) as composition:
        assert isinstance(
            (await composition.workflow_dependencies()).by_name["embedder"], adapter
        )


@pytest.mark.parametrize(
    ("backend", "adapter"),
    [("fake", FakeReranker), ("none", NoReranker), ("api", ApiReranker)],
)
async def test_each_rerank_backend_opens_its_adapter(
    settings_factory: SettingsFactory, backend: str, adapter: type
) -> None:
    settings = settings_factory(
        **(FAKE_BACKENDS | {"components__reranker__backend": backend}),
        components__reranker__api__url=RERANK_URL,
    )
    async with CompositionRoot(settings) as composition:
        assert isinstance(
            (await composition.workflow_dependencies()).by_name["reranker"], adapter
        )


@pytest.mark.parametrize(("backend", "adapter"), [("fake", FakeLLM), ("api", ApiLLM)])
async def test_each_generation_backend_opens_its_adapter(
    settings_factory: SettingsFactory, backend: str, adapter: type
) -> None:
    settings = settings_factory(
        **(FAKE_BACKENDS | {"components__llm__backend": backend})
    )
    async with CompositionRoot(settings) as composition:
        assert isinstance(
            (await composition.workflow_dependencies()).by_name["llm"], adapter
        )


# --- the api backends on the wire --------------------------------------------------


async def test_the_api_embedder_calls_its_endpoint(
    settings_factory: SettingsFactory,
) -> None:
    routes = RecordingModelRoutes()
    settings = settings_factory(
        **(FAKE_BACKENDS | {"components__embedder__backend": "api"}),
        components__embedder__api__url=EMBED_URL,
        components__embedder__api__api_key="emb-key",
        components__embedder__api__model="bge-small",
        components__embedder__api__batch_size=2,
        http__timeout_s=4.5,
    )
    transport = httpx.MockTransport(routes)
    composition = CompositionRoot(
        settings, vector_store=InMemoryVectorStore(), transport=transport
    )
    async with composition:
        embedder, _ = await composition.ingest_ports()
        assert await embedder.embed_documents(["a", "b", "c"]) == [[1.0], [1.0], [1.0]]

    assert routes.sent() == [
        (EMBED_URL, {"model": "bge-small", "input": ["a", "b"]}, "Bearer emb-key", 4.5),
        (EMBED_URL, {"model": "bge-small", "input": ["c"]}, "Bearer emb-key", 4.5),
    ]


async def test_the_api_reranker_calls_its_endpoint(
    settings_factory: SettingsFactory,
) -> None:
    routes = RecordingModelRoutes()
    settings = settings_factory(
        **(FAKE_BACKENDS | {"components__reranker__backend": "api"}),
        components__reranker__api__url=RERANK_URL,
        components__reranker__api__api_key="co-key",
        components__reranker__api__model="rerank-v3.5",
        http__timeout_s=4.5,
    )
    async with CompositionRoot(
        settings, transport=httpx.MockTransport(routes)
    ) as composition:
        reranker = (await composition.workflow_dependencies()).by_name["reranker"]
        [best] = await reranker.rerank("vacation", candidates(), top_n=1)

    assert (best.id, best.rerank_score) == ("a", 0.9)
    expected_body = {
        "model": "rerank-v3.5",
        "query": "vacation",
        "documents": ["vacation days", "expense receipts"],
        "top_n": 1,
    }
    assert routes.sent() == [(RERANK_URL, expected_body, "Bearer co-key", 4.5)]


async def test_the_api_llm_calls_its_endpoint_with_its_own_timeout(
    settings_factory: SettingsFactory,
) -> None:
    routes = RecordingModelRoutes()
    settings = settings_factory(
        **(FAKE_BACKENDS | {"components__llm__backend": "api"}),
        components__llm__api__url=CHAT_URL,
        components__llm__api__api_key="sk-key",
        components__llm__api__model="llama-3",
        components__llm__api__temperature=0.7,
        components__llm__http__timeout_s=90.0,
        http__timeout_s=4.5,
    )
    async with CompositionRoot(
        settings, transport=httpx.MockTransport(routes)
    ) as composition:
        llm = (await composition.workflow_dependencies()).by_name["llm"]
        messages = [{"role": "user", "content": "How many vacation days?"}]
        assert (await llm.generate(messages)).text == "Twenty-five days."

    expected_body = {"model": "llama-3", "messages": messages, "temperature": 0.7}
    assert routes.sent() == [(CHAT_URL, expected_body, "Bearer sk-key", 90.0)]


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool


@pytest.mark.parametrize("backend", ["api", "served"])
@pytest.mark.parametrize(
    ("method", "sends_a_response_format", "instructions"),
    [("json_schema", True, 0), ("format_instructions", False, 1)],
)
async def test_the_llm_takes_a_schema_the_way_its_block_names(
    settings_factory: SettingsFactory,
    backend: str,
    method: str,
    sends_a_response_format: bool,
    instructions: int,
) -> None:
    sent: list[dict[str, object]] = []

    def answer_ok(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        content = '```json\n{"ok": true}\n```' if instructions else '{"ok": true}'
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    block = {"url": CHAT_URL, "structured_output": method}
    settings = settings_factory(
        **(FAKE_BACKENDS | {"components__llm__backend": backend}),
        components={"llm": {"alias": "llama-3", backend: block}},
    )
    async with CompositionRoot(
        settings, transport=httpx.MockTransport(answer_ok)
    ) as composition:
        llm = (await composition.workflow_dependencies()).by_name["llm"]
        messages = [{"role": "user", "content": "Is it ok?"}]
        answer = await llm.generate(messages, output_schema=Verdict)

    assert json.loads(answer.text) == {"ok": True}
    [body] = sent
    assert ("response_format" in body) is sends_a_response_format
    assert len(body["messages"]) == 1 + instructions  # type: ignore[arg-type]


async def test_an_llm_that_sets_no_retries_keeps_the_default(
    settings_factory: SettingsFactory,
) -> None:
    attempts: list[str] = []

    def busy(request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        return httpx.Response(503, text="busy")

    settings = settings_factory(
        **(FAKE_BACKENDS | {"components__llm__backend": "api"}),
        http__retries=1,
        http__backoff_s=0.0,
    )
    async with CompositionRoot(
        settings, transport=httpx.MockTransport(busy)
    ) as composition:
        llm = (await composition.workflow_dependencies()).by_name["llm"]
        with pytest.raises(RagDependencyError, match="HTTP 503"):
            await llm.generate([{"role": "user", "content": "q"}])

    assert len(attempts) == 2  # the first attempt and http__retries=1
