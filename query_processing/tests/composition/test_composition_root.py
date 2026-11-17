"""``CompositionRoot``: settings in, ready ports out, every resource released on exit."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

import httpx
import pytest

from cpu_fakes import FakeCpuCounter
from query_processing.adapters.embedder.fake.fake_embedder import FakeEmbedder
from query_processing.adapters.embedder.local.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.adapters.reranker.fake.fake_reranker import FakeReranker
from query_processing.adapters.reranker.local.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from query_processing.adapters.vector_store.elasticsearch.elasticsearch_store import (
    ElasticsearchVectorStore,
)
from query_processing.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.errors import RagDependencyError
from query_processing.domain.passages import ScoredPassage
from rag_components import FAKE_BACKENDS
from sentence_transformers_fakes import FakeSentenceTransformersModule

SettingsFactory = Callable[..., WorkflowSettings]
EMBEDDER_WEIGHTS = "embedder:fixture-org/fixture-embedder"
UNREACHABLE_ES = "http://127.0.0.1:9"  # discard port: any connection would be refused
UNREACHABLE_STORE = {"components__vector_store__elasticsearch__url": UNREACHABLE_ES}
TOKEN_URL = "http://token.test/access-token"
PINNED_THREADS = {  # no calibration
    "components__embedder__local__threads": 1,
    "components__reranker__local__threads": 1,
}
# Too small for a second trial: calibration keeps the thread budget's share.
FIRST_CANDIDATES = {"calibration__budget_s": 1e-9}
EMBED_URL = "http://embedder/v3/embeddings"
RERANK_URL = "http://reranker/v3/rerank"


@pytest.fixture
def fake_weights(monkeypatch: pytest.MonkeyPatch) -> FakeSentenceTransformersModule:
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    # Under the default openvino runtime a first start exports each model into
    # QP_LOCAL_MODELS__OPENVINO_DIR; these tests are about the ports, so the models load
    # by id. The runtimes and the export: test_composition_root_backends.py.
    monkeypatch.setenv("QP_LOCAL_MODELS__RUNTIME", "torch")
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
    fake_settings: WorkflowSettings,
) -> None:
    async with CompositionRoot(fake_settings) as composition:
        deps = await composition.workflow_dependencies()

        assert isinstance(deps.by_name["embedder"], FakeEmbedder)
        assert isinstance(deps.by_name["vector_store"], InMemoryVectorStore)
        assert isinstance(deps.by_name["reranker"], FakeReranker)
        assert isinstance(deps.by_name["llm"], FakeLLM)
        # Every synthetic document fits one chunk, so passages == documents.
        store = deps.by_name["vector_store"]
        assert await store.ready() == (True, "200 passages in memory")


async def test_ports_open_once_and_are_reused(fake_settings: WorkflowSettings) -> None:
    async with CompositionRoot(fake_settings) as composition:
        first = await composition.workflow_dependencies()
        second = await composition.workflow_dependencies()
        store = (await composition.ingest_ports()).vector_store

    assert second.by_name["embedder"] is first.by_name["embedder"]
    assert second.by_name["vector_store"] is first.by_name["vector_store"] is store


# --- backends -------------------------------------------------------------------------


async def test_each_model_server_endpoint_fetches_its_own_domino_token(
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
        # both ports served by a model server of ours
        components__embedder__backend="served",
        components__reranker__backend="served",
        components__llm__backend="fake",
        domino__token_url=TOKEN_URL,
        http__timeout_s=7.5,
        components__embedder__served__url=EMBED_URL,
        components__reranker__served__url=RERANK_URL,
    )
    composition = CompositionRoot(
        settings,
        vector_store=InMemoryVectorStore(),
        transport=httpx.MockTransport(model_servers),
        on_domino=True,
    )
    async with composition:
        deps = await composition.workflow_dependencies()
        assert await deps.by_name["embedder"].embed_documents(["q"]) == [[1.0, 0.0]]
        candidate = ScoredPassage(id="a", text="t", retrieval_score=0.1)
        [ranked] = await deps.by_name["reranker"].rerank("q", [candidate], 1)
        assert ranked.rerank_score == 0.9

    # One BearerToken per endpoint: each has its own key, so its own cache.
    assert requested == [TOKEN_URL, EMBED_URL, TOKEN_URL, RERANK_URL]
    assert read_timeouts == {7.5}


async def test_api_embedder_batches_and_retries_as_configured(
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
        components__embedder__backend="api",
        components__embedder__api__batch_size=2,
        http__retries=0,
        components__embedder__api__url=EMBED_URL,
    )
    composition = CompositionRoot(
        settings,
        vector_store=InMemoryVectorStore(),
        transport=httpx.MockTransport(busy_then_ok),
    )
    async with composition:
        embedder, _ = await composition.ingest_ports()
        assert await embedder.embed_documents(["a", "b"]) == [[1.0], [1.0]]
        with pytest.raises(RagDependencyError, match="HTTP 503"):
            await embedder.embed_documents(["c"])

    assert sent == [["a", "b"], ["c"]]  # batches of 2; the 503 is not retried


async def test_local_models_use_the_configured_batch_sizes(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    batch_sizes = {
        "components__embedder__local__batch_size": 16,
        "components__reranker__local__batch_size": 8,
    }
    settings = settings_factory(**batch_sizes, **PINNED_THREADS)
    candidate = ScoredPassage(id="a", text="t", retrieval_score=0.1)
    async with CompositionRoot(settings) as composition:
        await (await composition.local_embedder("embedder")).embed_documents(["hello"])
        await (await composition.local_reranker("reranker")).rerank("q", [candidate], 1)

    assert fake_weights.batch_sizes == [("encode", 16), ("predict", 8)]


async def test_elasticsearch_store_opens_on_its_index_without_the_network(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        components__embedder__backend="fake",
        **UNREACHABLE_STORE,
        components__vector_store__elasticsearch__index="passages_v2",
    )
    async with CompositionRoot(settings) as composition:
        store = (await composition.ingest_ports()).vector_store
        assert isinstance(store, ElasticsearchVectorStore)
        assert store.index == "passages_v2"


async def test_fake_store_is_seeded_with_the_configured_document_count(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        **FAKE_BACKENDS, components__vector_store__fake__seed_documents=12
    )
    async with CompositionRoot(settings) as composition:
        store = (await composition.ingest_ports()).vector_store
        assert await store.count() == 12


# No module registers this format: the one way to make the api LLM fail to open.
UNKNOWN_GENERATION_FORMAT = {
    "components__llm__backend": "api",
    "components__llm__api__api_format": "nope",
}


async def test_the_llm_opens_before_any_weights_load(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    # local models, api generation
    settings = settings_factory(**UNREACHABLE_STORE, **UNKNOWN_GENERATION_FORMAT)
    async with CompositionRoot(settings) as composition:
        with pytest.raises(RagDependencyError, match="llm unavailable"):
            await composition.workflow_dependencies()

    assert (
        fake_weights.loaded == []
    )  # a bad generation setting fails before the slow part


# --- each caller gets only what it asks for -------------------------------------------


async def test_ingest_ports_leave_the_llm_shut(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(**{**FAKE_BACKENDS, **UNKNOWN_GENERATION_FORMAT})
    async with CompositionRoot(settings) as composition:
        _, store = await composition.ingest_ports()  # the LLM is never opened
        assert await store.count() == 200  # the default seed size

        expected = "llm unavailable: could not open backend 'api'"
        with pytest.raises(RagDependencyError, match=expected):
            await composition.workflow_dependencies()


async def test_ingest_ports_load_only_the_embedder_weights(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    # local embedder and reranker
    settings = settings_factory(**UNREACHABLE_STORE, **PINNED_THREADS)
    async with CompositionRoot(settings) as composition:
        await composition.ingest_ports()

    assert fake_weights.loaded == [EMBEDDER_WEIGHTS]


async def test_local_models_ignore_the_configured_backend(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    settings = settings_factory(  # both backends served
        components__embedder__backend="served",
        components__reranker__backend="served",
        components__embedder__served__url=EMBED_URL,
        components__reranker__served__url=RERANK_URL,
        **PINNED_THREADS,
    )
    async with CompositionRoot(settings) as composition:
        embedder = await composition.local_embedder("embedder")

        assert isinstance(embedder, SentenceTransformerEmbedder)
        assert await embedder.embed_documents(["hello"]) == [[1.0, 0.0]]
        assert fake_weights.loaded == [EMBEDDER_WEIGHTS]
        reranker = await composition.local_reranker("reranker")
        assert isinstance(reranker, CrossEncoderReranker)


# --- ports handed in ------------------------------------------------------------------------


async def test_a_supplied_port_is_used_as_is(fake_settings: WorkflowSettings) -> None:
    embedder, empty_store = FakeEmbedder(dim=4), InMemoryVectorStore()
    composition = CompositionRoot(
        fake_settings, embedder=embedder, vector_store=empty_store
    )
    async with composition:
        deps = await composition.workflow_dependencies()

        assert deps.by_name["embedder"] is embedder
        assert deps.by_name["vector_store"] is empty_store
        assert await empty_store.count() == 0  # a supplied store is never seeded
        assert await composition.local_embedder("embedder") is embedder


async def test_fake_store_is_seeded_with_the_supplied_embedder(
    fake_settings: WorkflowSettings,
) -> None:
    embedder = FakeEmbedder(dim=4)
    async with CompositionRoot(fake_settings, embedder=embedder) as composition:
        store = (await composition.ingest_ports()).vector_store
        # A store seeded by any other embedder would hold 16-wide vectors and raise.
        hits = await store.query(embedder.embed_text("vacation days"), top_k=3)

    assert len(hits) == 3


# --- lifecycle ------------------------------------------------------------------------


async def test_ports_cannot_be_requested_outside_the_block(
    fake_settings: WorkflowSettings,
) -> None:
    composition = CompositionRoot(fake_settings)
    with pytest.raises(
        RuntimeError, match=r"outside 'async with CompositionRoot\(\.\.\.\)'"
    ):
        await composition.workflow_dependencies()


async def test_leaving_the_block_releases_the_pool_and_the_http_client(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    transport = ClosingRecorderTransport()
    settings = settings_factory(
        components__reranker__backend="served",
        components__reranker__served__url=RERANK_URL,
        components__llm__backend="fake",
    )
    composition = CompositionRoot(
        settings, vector_store=InMemoryVectorStore(), transport=transport
    )
    async with composition:
        deps = await composition.workflow_dependencies()
        assert not transport.closed

    assert transport.closed
    with pytest.raises(RuntimeError, match="after shutdown"):
        await deps.by_name["embedder"].embed_documents(["the model pool is gone"])


# --- thread budget ----------------------------------------------------------------------


def thread_budget_line(caplog: pytest.LogCaptureFixture) -> dict[str, object]:
    """The one ``thread_budget`` JSON line CompositionRoot logged."""
    lines = [
        json.loads(r.message) for r in caplog.records if "thread_budget" in r.message
    ]
    [line] = lines
    return line


async def open_local_models(
    composition: CompositionRoot,
) -> tuple[SentenceTransformerEmbedder, CrossEncoderReranker]:
    async with composition:
        embedder, reranker = (
            await composition.local_embedder("embedder"),
            await composition.local_reranker("reranker"),
        )
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert isinstance(reranker, CrossEncoderReranker)
    return embedder, reranker


async def test_local_model_scopes_carry_the_thread_budget(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    container = FakeCpuCounter(8, "container")
    settings = settings_factory(**FIRST_CANDIDATES)
    composition = CompositionRoot(settings, container_cpus=container)
    with caplog.at_level("INFO", logger="query_processing"):
        embedder, reranker = await open_local_models(composition)

    assert (embedder.scope.threads, reranker.scope.threads) == (2, 2)
    assert thread_budget_line(caplog) == {
        "event": "thread_budget",
        "cpus": 8,
        "source": "container",
        "runtime": "torch",
        "workers": 4,
        "embedder_threads": 2,
        "reranker_threads": 2,
    }


async def test_a_thread_override_replaces_one_share(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    settings = settings_factory(
        components__embedder__local__threads=1, **FIRST_CANDIDATES
    )
    composition = CompositionRoot(
        settings, container_cpus=FakeCpuCounter(8, "container")
    )
    embedder, reranker = await open_local_models(composition)

    assert (embedder.scope.threads, reranker.scope.threads) == (1, 2)


async def test_model_cpus_win_over_every_counter(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    container = FakeCpuCounter(8, "container")
    settings = settings_factory(
        local_models__cpus=3,
        domino_api_proxy="http://proxy",
        domino_hardware_tier_id="t",
    )
    composition = CompositionRoot(
        settings, container_cpus=container
    )  # no transport: no HTTP
    with caplog.at_level("INFO", logger="query_processing"):
        await open_local_models(composition)

    line = thread_budget_line(caplog)
    assert (line["source"], line["cpus"]) == ("QP_LOCAL_MODELS__CPUS", 3)
    assert line["workers"] == 3
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
    composition = CompositionRoot(
        settings, container_cpus=container, transport=transport
    )
    with caplog.at_level("INFO", logger="query_processing"):
        embedder, _ = await open_local_models(composition)

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
    composition = CompositionRoot(
        settings, container_cpus=container, transport=transport
    )
    embedder, _ = await open_local_models(composition)

    assert container.asked == 1
    assert embedder.scope.threads == 1


async def test_fake_models_never_work_out_a_budget(
    fake_settings: WorkflowSettings,
) -> None:
    container = FakeCpuCounter(8, "container")
    async with CompositionRoot(fake_settings, container_cpus=container) as composition:
        await composition.workflow_dependencies()

    assert container.asked == 0
