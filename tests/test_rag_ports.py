"""``RagPorts``: settings in, ready ports out, every resource released on exit."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from types import ModuleType

import httpx
import numpy as np
import pytest

from cpu_fakes import FakeCpuCounter
from rag_load_test.adapters.elasticsearch_store import ElasticsearchVectorStore
from rag_load_test.adapters.fakes import (
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
    InMemoryVectorStore,
)
from rag_load_test.adapters.local_models import (
    CrossEncoderReranker,
    NoReranker,
    SentenceTransformerEmbedder,
)
from rag_load_test.adapters.rag_ports import RagPorts
from rag_load_test.config.settings import RagSettings
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.passages import ScoredPassage

SettingsFactory = Callable[..., RagSettings]
EMBEDDER_WEIGHTS = "embedder:BAAI/bge-small-en-v1.5"
UNREACHABLE_ES = "http://127.0.0.1:9"  # discard port: any connection would be refused
TOKEN_URL = "http://token.test/access-token"


class FakeSentenceTransformersModule(ModuleType):
    """Stands in for the ``sentence_transformers`` package; records which weights load."""

    def __init__(self) -> None:
        super().__init__("sentence_transformers")
        self.loaded: list[str] = []
        self.batch_sizes: list[tuple[str, int]] = []
        loaded, batch_sizes = self.loaded, self.batch_sizes

        class SentenceTransformer:
            def __init__(self, name: str, device: str) -> None:
                del device  # passed by keyword, as to the real class
                loaded.append(f"embedder:{name}")

            def encode(
                self, texts: list[str], *, batch_size: int, normalize_embeddings: bool
            ) -> np.ndarray:
                batch_sizes.append(("encode", batch_size))
                return np.array([[1.0, 0.0] for _ in texts])

        class CrossEncoder:
            def __init__(self, name: str, device: str) -> None:
                del device  # passed by keyword, as to the real class
                loaded.append(f"reranker:{name}")

            def predict(
                self, pairs: list[tuple[str, str]], batch_size: int
            ) -> list[float]:
                batch_sizes.append(("predict", batch_size))
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
    read_timeouts: set[float] = set()

    def model_servers(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        read_timeouts.add(request.extensions["timeout"]["read"])
        if str(request.url) == TOKEN_URL:
            return httpx.Response(200, text="run-token")
        assert request.headers["Authorization"] == "Bearer run-token"
        if request.url.path == "/v3/embeddings":
            row = {"index": 0, "embedding": [1.0, 0.0]}
            return httpx.Response(200, json={"data": [row]})
        result = {"index": 0, "relevance_score": 0.9}
        return httpx.Response(200, json={"results": [result]})

    settings = settings_factory(
        topology="split-all",  # both ports on a model server
        llm_backend="fake",
        ovms_auth="domino",
        domino_token_url=TOKEN_URL,
        http_timeout_s=7.5,
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
        TOKEN_URL,
        "http://embedder/v3/embeddings",
        "http://reranker/v3/rerank",
    ]
    assert read_timeouts == {7.5}


async def test_ovms_embedder_batches_and_retries_as_configured(
    settings_factory: SettingsFactory,
) -> None:
    sent: list[list[str]] = []

    def busy_then_ok(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        sent.append(inputs)
        if inputs == ["c"]:
            return httpx.Response(503, text="busy")
        rows = [{"index": i, "embedding": [1.0]} for i in range(len(inputs))]
        return httpx.Response(200, json={"data": rows})

    settings = settings_factory(
        embedder_backend="ovms",
        topology="split-all",
        embedder_batch_size=2,
        http_retries=0,
        ovms_embeddings_url="http://embedder",
    )
    ports = RagPorts(
        settings,
        vector_store=InMemoryVectorStore(),
        transport=httpx.MockTransport(busy_then_ok),
    )
    async with ports:
        embedder, _ = await ports.ingest_ports()
        assert await embedder.embed(["a", "b"]) == [[1.0], [1.0]]
        with pytest.raises(RagDependencyError, match="HTTP 503"):
            await embedder.embed(["c"])

    assert sent == [["a", "b"], ["c"]]  # batches of 2; the 503 is not retried


async def test_local_models_use_the_configured_batch_sizes(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    settings = settings_factory(embedder_batch_size=16, reranker_batch_size=8)
    candidate = ScoredPassage(id="a", text="t", retrieval_score=0.1)
    async with RagPorts(settings) as ports:
        await (await ports.local_embedder()).embed(["hello"])
        await (await ports.local_reranker()).rerank("q", [candidate], 1)

    assert fake_weights.batch_sizes == [("encode", 16), ("predict", 8)]


async def test_elasticsearch_store_uses_the_configured_index(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        embedder_backend="fake", es_url=UNREACHABLE_ES, es_index="passages_v2"
    )
    async with RagPorts(settings) as ports:
        store = (await ports.ingest_ports()).vector_store
        assert isinstance(store, ElasticsearchVectorStore)
        assert store.index == "passages_v2"


async def test_fake_store_is_seeded_with_the_configured_document_count(
    fake_settings: RagSettings,
) -> None:
    settings = fake_settings.model_copy(update={"fake_store_seed_documents": 12})
    async with RagPorts(settings) as ports:
        store = (await ports.ingest_ports()).vector_store
        assert await store.count() == 12


async def test_elasticsearch_store_opens_without_the_network(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(embedder_backend="fake", es_url=UNREACHABLE_ES)
    async with RagPorts(settings) as ports:
        store = (await ports.ingest_ports()).vector_store
        assert isinstance(store, ElasticsearchVectorStore)


async def test_the_chat_model_opens_before_any_weights_load(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = settings_factory(es_url=UNREACHABLE_ES)  # local models, OpenAI chat
    async with RagPorts(settings) as ports:
        with pytest.raises(RagDependencyError, match="chat_model unavailable"):
            await ports.workflow_dependencies()

    assert fake_weights.loaded == []  # a missing key fails before the slow part


# --- each caller gets only what it asks for -------------------------------------------


async def test_ingest_ports_leave_the_chat_model_shut(
    fake_settings: RagSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = fake_settings.model_copy(update={"llm_backend": "openai"})
    async with RagPorts(settings) as ports:
        _, store = await ports.ingest_ports()  # no OpenAI client: no key needed
        assert await store.count() == 200  # the default seed size

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
    settings = settings_factory(topology="split-all")  # both backends ovms
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

        assert deps.embedder is embedder
        assert deps.vector_store is empty_store
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
    settings = settings_factory(topology="split-reranker", llm_backend="fake")
    ports = RagPorts(settings, vector_store=InMemoryVectorStore(), transport=transport)
    async with ports:
        deps = await ports.workflow_dependencies()
        assert not transport.closed

    assert transport.closed
    with pytest.raises(RuntimeError, match="after shutdown"):
        await deps.embedder.embed(["the model pool is gone"])


# --- thread budget ----------------------------------------------------------------------


def thread_budget_line(caplog: pytest.LogCaptureFixture) -> dict[str, object]:
    """The one ``thread_budget`` JSON line RagPorts logged."""
    lines = [
        json.loads(r.message) for r in caplog.records if "thread_budget" in r.message
    ]
    [line] = lines
    return line


async def open_local_models(
    ports: RagPorts,
) -> tuple[SentenceTransformerEmbedder, CrossEncoderReranker]:
    async with ports:
        embedder, reranker = await ports.local_embedder(), await ports.local_reranker()
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert isinstance(reranker, CrossEncoderReranker)
    return embedder, reranker


async def test_local_model_scopes_carry_the_thread_budget(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    container = FakeCpuCounter(8, "container")
    ports = RagPorts(settings_factory(), container_cpus=container)
    with caplog.at_level("INFO", logger="rag_load_test"):
        embedder, reranker = await open_local_models(ports)

    assert (embedder.scope.threads, reranker.scope.threads) == (2, 2)
    assert thread_budget_line(caplog) == {
        "event": "thread_budget",
        "cpus": 8,
        "source": "container",
        "workers": 4,
        "embedder_threads": 2,
        "reranker_threads": 2,
    }


async def test_a_thread_override_replaces_one_share(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    settings = settings_factory(embedder_threads=1)
    ports = RagPorts(settings, container_cpus=FakeCpuCounter(8, "container"))
    embedder, reranker = await open_local_models(ports)

    assert (embedder.scope.threads, reranker.scope.threads) == (1, 2)


async def test_model_cpus_win_over_every_counter(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    container = FakeCpuCounter(8, "container")
    settings = settings_factory(
        model_cpus=3, domino_api_proxy="http://proxy", domino_hardware_tier_id="t"
    )
    ports = RagPorts(settings, container_cpus=container)  # no transport: no HTTP
    with caplog.at_level("INFO", logger="rag_load_test"):
        await open_local_models(ports)

    line = thread_budget_line(caplog)
    assert (line["source"], line["cpus"], line["workers"]) == ("RAG_MODEL_CPUS", 3, 3)
    assert container.asked == 0


async def test_on_domino_the_hardware_tier_is_asked(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    asked: list[str] = []

    def domino(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        resources = {"cores": 6.0}
        return httpx.Response(200, json={"hardwareTier": {"resources": resources}})

    container = FakeCpuCounter(8, "container")
    settings = settings_factory(
        domino_api_proxy="http://proxy", domino_hardware_tier_id="tier-9"
    )
    transport = httpx.MockTransport(domino)
    ports = RagPorts(settings, container_cpus=container, transport=transport)
    with caplog.at_level("INFO", logger="rag_load_test"):
        embedder, _ = await open_local_models(ports)

    assert asked == ["/api/hardwaretiers/v1/hardwaretiers/tier-9"]
    line = thread_budget_line(caplog)
    assert (line["source"], line["cpus"], line["workers"]) == (
        "domino_hardware_tier",
        6.0,
        4,
    )
    assert embedder.scope.threads == 1
    assert container.asked == 0


async def test_when_the_tier_cannot_say_the_container_answers(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    def forbidden(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="no ViewHardwareTiers")

    container = FakeCpuCounter(2, "container")
    settings = settings_factory(
        domino_api_proxy="http://proxy", domino_hardware_tier_id="tier-9"
    )
    transport = httpx.MockTransport(forbidden)
    ports = RagPorts(settings, container_cpus=container, transport=transport)
    embedder, _ = await open_local_models(ports)

    assert container.asked == 1
    assert embedder.scope.threads == 1


async def test_fake_models_never_work_out_a_budget(
    fake_settings: RagSettings,
) -> None:
    container = FakeCpuCounter(8, "container")
    async with RagPorts(fake_settings, container_cpus=container) as ports:
        await ports.workflow_dependencies()

    assert container.asked == 0
