"""``RagPorts``: settings in, ready ports out, every resource released on exit."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType

import httpx
import numpy as np
import pytest

from rag_load_test.adapters import (
    CrossEncoderReranker,
    ElasticsearchVectorStore,
    NoReranker,
    SentenceTransformerEmbedder,
)
from rag_load_test.fakes import (
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
    InMemoryVectorStore,
)
from rag_load_test.models import RagDependencyError, ScoredPassage
from rag_load_test.ovms import DOMINO_TOKEN_URL
from rag_load_test.rag_ports import FAKE_STORE_SEED_DOCUMENTS, RagPorts
from rag_load_test.settings import RagSettings

SettingsFactory = Callable[..., RagSettings]
EMBEDDER_WEIGHTS = "embedder:BAAI/bge-small-en-v1.5"
UNREACHABLE_ES = "http://127.0.0.1:9"  # discard port: any connection would be refused


class FakeSentenceTransformersModule(ModuleType):
    """Stands in for the ``sentence_transformers`` package; records which weights load."""

    def __init__(self) -> None:
        super().__init__("sentence_transformers")
        self.loaded: list[str] = []
        loaded = self.loaded

        class SentenceTransformer:
            def __init__(self, name: str, device: str) -> None:
                loaded.append(f"embedder:{name}")

            def encode(
                self, texts: list[str], *, batch_size: int, normalize_embeddings: bool
            ) -> np.ndarray:
                return np.array([[1.0, 0.0] for _ in texts])

        class CrossEncoder:
            def __init__(self, name: str, device: str) -> None:
                loaded.append(f"reranker:{name}")

            def predict(
                self, pairs: list[tuple[str, str]], batch_size: int
            ) -> list[float]:
                return [0.5 for _ in pairs]

        self.SentenceTransformer = SentenceTransformer
        self.CrossEncoder = CrossEncoder


@pytest.fixture
def fake_weights(monkeypatch: pytest.MonkeyPatch) -> FakeSentenceTransformersModule:
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


class ClosingRecorderTransport(httpx.MockTransport):
    """An HTTP transport that answers 200 and records being closed."""

    def __init__(self) -> None:
        super().__init__(lambda request: httpx.Response(200))
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


# --- the workflow app's entry point ------------------------------------------------


async def test_all_fake_settings_open_four_ports_over_a_seeded_store(
    fake_settings: RagSettings,
) -> None:
    async with RagPorts(fake_settings) as ports:
        deps = await ports.workflow_dependencies()

        assert isinstance(deps.embedder, FakeEmbedder)
        assert isinstance(deps.vector_store, InMemoryVectorStore)
        assert isinstance(deps.reranker, FakeReranker)
        assert isinstance(deps.chat_model, FakeChatModel)
        # Every synthetic document fits one chunk, so passages == documents.
        assert FAKE_STORE_SEED_DOCUMENTS == 200
        assert await deps.vector_store.ready() == (True, "200 passages in memory")


async def test_ports_open_once_and_are_reused(fake_settings: RagSettings) -> None:
    async with RagPorts(fake_settings) as ports:
        first = await ports.workflow_dependencies()
        second = await ports.workflow_dependencies()

        assert second.embedder is first.embedder
        assert second.vector_store is first.vector_store
        assert (await ports.ingest_ports()).vector_store is first.vector_store


# --- backends -------------------------------------------------------------------------


async def test_none_reranker_backend_keeps_retrieval_order(
    settings_factory: SettingsFactory, fake_settings: RagSettings
) -> None:
    settings = fake_settings.model_copy(update={"reranker_backend": "none"})
    async with RagPorts(settings) as ports:
        assert isinstance((await ports.workflow_dependencies()).reranker, NoReranker)


async def test_both_model_server_adapters_share_one_domino_token(
    settings_factory: SettingsFactory,
) -> None:
    requested: list[str] = []

    def model_servers(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if str(request.url) == DOMINO_TOKEN_URL:
            return httpx.Response(200, text="run-token")
        assert request.headers["Authorization"] == "Bearer run-token"
        if request.url.path == "/v3/embeddings":
            row = {"index": 0, "embedding": [1.0, 0.0]}
            return httpx.Response(200, json={"data": [row]})
        result = {"index": 0, "relevance_score": 0.9}
        return httpx.Response(200, json={"results": [result]})

    settings = settings_factory(
        embedder_backend="ovms",
        reranker_backend="ovms",
        llm_backend="fake",
        ovms_auth="domino",
        ovms_embeddings_url="http://embedder",
        ovms_rerank_url="http://reranker",
    )
    ports = RagPorts(
        settings,
        vector_store=InMemoryVectorStore(),
        transport=httpx.MockTransport(model_servers),
    )
    async with ports:
        deps = await ports.workflow_dependencies()
        assert await deps.embedder.embed(["q"]) == [[1.0, 0.0]]
        candidate = ScoredPassage(id="a", text="t", retrieval_score=0.1)
        [ranked] = await deps.reranker.rerank("q", [candidate], 1)
        assert ranked.rerank_score == 0.9

    assert requested == [
        DOMINO_TOKEN_URL,
        "http://embedder/v3/embeddings",
        "http://reranker/v3/rerank",
    ]


async def test_elasticsearch_store_opens_and_closes_without_the_network(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(embedder_backend="fake", es_url=UNREACHABLE_ES)
    async with RagPorts(settings) as ports:
        store = (await ports.ingest_ports()).vector_store
        assert isinstance(store, ElasticsearchVectorStore)


# --- each caller gets only what it asks for -------------------------------------------


async def test_ingest_ports_leave_the_chat_model_shut(
    fake_settings: RagSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = fake_settings.model_copy(update={"llm_backend": "openai"})
    async with RagPorts(settings) as ports:
        _, store = await ports.ingest_ports()  # no OpenAI client: no key needed
        assert await store.count() == FAKE_STORE_SEED_DOCUMENTS

        expected = "chat_model unavailable: could not open backend 'openai'"
        with pytest.raises(RagDependencyError, match=expected):
            await ports.workflow_dependencies()


async def test_ingest_ports_load_only_the_embedder_weights(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    settings = settings_factory(es_url=UNREACHABLE_ES)  # local embedder and reranker
    async with RagPorts(settings) as ports:
        await ports.ingest_ports()

    assert fake_weights.loaded == [EMBEDDER_WEIGHTS]


async def test_local_models_ignore_the_configured_backend(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    settings = settings_factory(embedder_backend="ovms", reranker_backend="ovms")
    async with RagPorts(settings) as ports:
        embedder = await ports.local_embedder()

        assert isinstance(embedder, SentenceTransformerEmbedder)
        assert await embedder.embed(["hello"]) == [[1.0, 0.0]]
        assert fake_weights.loaded == [EMBEDDER_WEIGHTS]
        assert isinstance(await ports.local_reranker(), CrossEncoderReranker)


# --- ports handed in ------------------------------------------------------------------


async def test_a_supplied_port_is_used_as_given(fake_settings: RagSettings) -> None:
    embedder, empty_store = FakeEmbedder(dim=4), InMemoryVectorStore()
    ports = RagPorts(fake_settings, embedder=embedder, vector_store=empty_store)
    async with ports:
        deps = await ports.workflow_dependencies()

        assert deps.embedder is embedder and deps.vector_store is empty_store
        assert await empty_store.count() == 0  # a supplied store is never seeded
        assert await ports.local_embedder() is embedder


async def test_fake_store_is_seeded_with_the_supplied_embedder(
    fake_settings: RagSettings,
) -> None:
    embedder = FakeEmbedder(dim=4)
    async with RagPorts(fake_settings, embedder=embedder) as ports:
        store = (await ports.ingest_ports()).vector_store
        # A store seeded by any other embedder would hold 16-wide vectors and raise.
        hits = await store.query(embedder.embed_text("vacation days"), top_k=3)

    assert len(hits) == 3


# --- lifecycle ------------------------------------------------------------------------


async def test_ports_cannot_be_requested_outside_the_block(
    fake_settings: RagSettings,
) -> None:
    ports = RagPorts(fake_settings)
    with pytest.raises(RuntimeError, match=r"outside 'async with RagPorts\(\.\.\.\)'"):
        await ports.workflow_dependencies()


async def test_leaving_the_block_releases_the_pool_and_the_http_client(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    transport = ClosingRecorderTransport()
    settings = settings_factory(reranker_backend="ovms", llm_backend="fake")
    ports = RagPorts(settings, vector_store=InMemoryVectorStore(), transport=transport)
    async with ports:
        deps = await ports.workflow_dependencies()
        assert not transport.closed

    assert transport.closed
    with pytest.raises(RuntimeError, match="after shutdown"):
        await deps.embedder.embed(["the model pool is gone"])
