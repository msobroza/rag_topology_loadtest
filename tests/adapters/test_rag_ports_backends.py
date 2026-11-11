"""``RagPorts`` over the backends beyond local, OVMS and fake: Chroma, the hosted
model APIs, the in-process runtime and the chat temperature. Settings in, the
right adapter out."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from rag_load_test.adapters.chat_model.openai_chat import OpenAIChatModel
from rag_load_test.adapters.hosted_models.openai_embedder import OpenAIEmbedder
from rag_load_test.adapters.hosted_models.rerank_api import RerankApiReranker
from rag_load_test.adapters.local_models.local_model import local_model
from rag_load_test.adapters.rag_ports import RagPorts
from rag_load_test.adapters.vector_store.chroma_store import ChromaVectorStore
from rag_load_test.config.settings import RagSettings
from rag_load_test.domain.passages import Passage, ScoredPassage
from sentence_transformers_fakes import FakeSentenceTransformersModule

SettingsFactory = Callable[..., RagSettings]


class RecordingRerankRoute(httpx.MockTransport):
    """A rerank route that keeps the first candidate and records each request."""

    def __init__(self) -> None:
        super().__init__(self._answer)
        self.requests: list[httpx.Request] = []

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        results = [{"index": 0, "relevance_score": 0.9}]
        return httpx.Response(200, json={"results": results})


def candidates() -> list[ScoredPassage]:
    return [
        ScoredPassage(id="a", text="vacation days", retrieval_score=0.5),
        ScoredPassage(id="b", text="expense receipts", retrieval_score=0.4),
    ]


# --- Chroma ----------------------------------------------------------------------


async def test_the_chroma_backend_opens_the_configured_embedded_collection(
    settings_factory: SettingsFactory, tmp_path: Path
) -> None:
    settings = settings_factory(
        embedder_backend="fake",
        vector_store_backend="chroma",
        chroma_path=str(tmp_path / "chroma"),
        chroma_collection="passages_under_test",
    )

    async with RagPorts(settings) as ports:
        embedder, store = await ports.ingest_ports()
        passage = Passage(id="p1", text="Employees accrue vacation days")
        await store.upsert([passage], await embedder.embed([passage.text]))

        assert isinstance(store, ChromaVectorStore)
        assert await store.ready() == (True, "1 passages in passages_under_test")
        assert [hit.id for hit in await store.lexical_query("vacation", 5)] == ["p1"]
    assert (tmp_path / "chroma").is_dir()  # embedded: written where the settings say


# --- hosted embedder -----------------------------------------------------------------


async def test_the_openai_embedder_backend_opens_and_closes_an_sdk_client(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        embedder_backend="openai",
        vector_store_backend="fake",
        openai_api_key="sk-test",
        openai_embedding_model="text-embedding-test",
    )

    async with RagPorts(settings) as ports:
        embedder = await ports._embedder()
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.model_name == "openai:text-embedding-test"
        client = embedder._client
        assert not client.is_closed()
    assert client.is_closed()


# --- hosted rerankers ----------------------------------------------------------------


async def test_the_cohere_reranker_backend_posts_to_cohere_with_its_key(
    settings_factory: SettingsFactory,
) -> None:
    route = RecordingRerankRoute()
    settings = settings_factory(reranker_backend="cohere", cohere_api_key="co-key")

    async with RagPorts(settings, transport=route) as ports:
        reranker = await ports._reranker()
        best = await reranker.rerank("vacation", candidates(), top_n=1)

    assert isinstance(reranker, RerankApiReranker)
    assert [passage.id for passage in best] == ["a"]
    [request] = route.requests
    assert str(request.url) == "https://api.cohere.com/v2/rerank"
    assert request.headers["Authorization"] == "Bearer co-key"
    assert json.loads(request.content)["model"] == "rerank-v3.5"


async def test_the_openai_reranker_backend_posts_to_the_gateways_rerank_route(
    settings_factory: SettingsFactory,
) -> None:
    route = RecordingRerankRoute()
    settings = settings_factory(
        reranker_backend="openai",
        openai_base_url="https://gateway.example/v1",
        openai_rerank_model="rerank-model",
        openai_api_key="sk-key",
    )

    async with RagPorts(settings, transport=route) as ports:
        reranker = await ports._reranker()
        await reranker.rerank("vacation", candidates(), top_n=1)

    [request] = route.requests
    assert str(request.url) == "https://gateway.example/v1/rerank"
    assert request.headers["Authorization"] == "Bearer sk-key"
    assert json.loads(request.content)["model"] == "rerank-model"


# --- in-process runtime ---------------------------------------------------------------


@pytest.fixture
def fake_weights(monkeypatch: pytest.MonkeyPatch) -> FakeSentenceTransformersModule:
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def local_settings(
    settings_factory: SettingsFactory, **overrides: object
) -> RagSettings:
    """In-process models on a known CPU budget; everything else fake."""
    return settings_factory(
        vector_store_backend="fake", llm_backend="fake", model_cpus=4, **overrides
    )


@pytest.mark.parametrize("runtime", ["openvino", "torch"])
async def test_local_models_load_with_the_arguments_of_the_configured_runtime(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
    runtime: str,
) -> None:
    settings = local_settings(
        settings_factory,
        local_model_runtime=runtime,
        openvino_models_dir=str(tmp_path / "nothing-exported"),
    )

    async with RagPorts(settings) as ports:
        await ports.local_embedder()
        await ports.local_reranker()

    # 4 CPUs over 4 workers: one thread per call, for both models.
    described = local_model(settings.embedder_model, runtime=runtime, models_dir="")
    assert fake_weights.load_arguments == [described.load_arguments(1)] * 2
    assert fake_weights.loaded == [
        f"embedder:{settings.embedder_model}",
        f"reranker:{settings.reranker_model}",
    ]


async def test_the_openvino_runtime_loads_a_model_from_its_exported_folder(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
) -> None:
    exported = tmp_path / "snowflake-arctic-embed-l-v2.0"
    exported.mkdir()
    settings = local_settings(settings_factory, openvino_models_dir=str(tmp_path))

    async with RagPorts(settings) as ports:
        embedder = await ports.local_embedder()
        await ports.local_reranker()

        assert await embedder.ready() == (True, settings.embedder_model)
    assert fake_weights.loaded == [
        f"embedder:{exported}",  # exported: loaded from the folder
        f"reranker:{settings.reranker_model}",  # not exported: the id
    ]


async def test_the_torch_runtime_ignores_an_exported_folder(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    tmp_path: Path,
) -> None:
    (tmp_path / "snowflake-arctic-embed-l-v2.0").mkdir()
    settings = local_settings(
        settings_factory, local_model_runtime="torch", openvino_models_dir=str(tmp_path)
    )

    async with RagPorts(settings) as ports:
        await ports.local_embedder()

    assert fake_weights.loaded == [f"embedder:{settings.embedder_model}"]


# --- chat temperature ------------------------------------------------------------------


async def test_the_chat_model_gets_the_configured_temperature(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # the SDK refuses to start without
    settings = settings_factory(
        embedder_backend="fake",
        reranker_backend="fake",
        vector_store_backend="fake",
        llm_temperature=0.7,
    )

    async with RagPorts(settings) as ports:
        deps = await ports.workflow_dependencies()

    assert isinstance(deps.chat_model, OpenAIChatModel)
    assert deps.chat_model.temperature == 0.7
