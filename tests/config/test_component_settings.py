"""Each component's settings in the workflow file's shape, derived from the flat ones."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from rag_load_test.config.component_settings import (
    ChromaSettings,
    ElasticsearchSettings,
    EmbedderEndpointSettings,
    EndpointSettings,
    FakeStoreSettings,
    HttpCallSettings,
    InProcessModelSettings,
    LlmEndpointSettings,
    RerankerLocalSettings,
    component_settings_of,
)
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.api_endpoint import ApiEndpoint

SettingsFactory = Callable[..., RagSettings]
EMBED_URL = "http://embedder:8002/v3/embeddings"
RERANK_URL = "http://reranker:8001/v3/rerank"


def test_the_defaults_take_the_workflow_file_shape(
    settings_factory: SettingsFactory,
) -> None:
    components = component_settings_of(settings_factory())

    embedder = components.embedder
    assert (embedder.backend, embedder.query_prefix) == ("local", "query: ")
    assert embedder.source_id == "Snowflake/snowflake-arctic-embed-l-v2.0"
    assert embedder.local == InProcessModelSettings(batch_size=64, threads=0)
    assert embedder.api == EmbedderEndpointSettings(
        url="",
        api_format="openai",
        api_key="",
        model="snowflake-arctic-embed-l-v2.0",
        batch_size=64,
    )
    reranker = components.reranker
    assert reranker.backend == "local"
    assert reranker.local == RerankerLocalSettings(
        batch_size=0, threads=0, calibration_passages=20, calibration_top_n=10
    )
    assert reranker.api.api_format == "cohere"
    llm = components.llm
    assert llm.backend == "api"
    assert llm.api == LlmEndpointSettings(
        url="https://api.openai.com/v1/chat/completions",
        api_format="openai",
        api_key="",
        model="gpt-4o-mini",
        temperature=0.0,
    )
    assert llm.http == HttpCallSettings(timeout_s=120.0, retries=2, backoff_s=0.2)
    assert components.vector_store.backend == "elasticsearch"


def test_every_flat_value_reaches_its_component_block(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        topology="split-all",
        embedding_api_url=EMBED_URL,
        embedding_api_key="domino",
        embedding_batch_size=8,
        embedding_threads=3,
        embedding_query_prefix="",
        rerank_api_url=RERANK_URL,
        rerank_batch_size=4,
        rerank_threads=2,
        top_k_retrieve=12,
        top_k_rerank=5,
        generation_timeout_s=9.0,
        http_retries=0,
        http_backoff_s=0.0,
        generation_temperature=0.5,
        es_url="http://es:9200",
        es_index="passages_v2",
        es_api_key="es-key",
        es_text_analyzer="french",
        chroma_url="http://chroma:8000",
        chroma_path="here",
        chroma_collection="passages",
        fake_store_seed_documents=12,
    )

    components = component_settings_of(settings)

    embedder, reranker = components.embedder, components.reranker
    assert (embedder.backend, reranker.backend) == ("api", "api")
    assert embedder.query_prefix == ""
    assert embedder.local == InProcessModelSettings(batch_size=8, threads=3)
    assert (embedder.api.url, embedder.api.api_key) == (EMBED_URL, "domino")
    assert embedder.api.batch_size == 8
    assert reranker.local == RerankerLocalSettings(
        batch_size=4, threads=2, calibration_passages=12, calibration_top_n=5
    )
    assert reranker.api.url == RERANK_URL
    assert components.llm.http == HttpCallSettings(
        timeout_s=9.0, retries=0, backoff_s=0.0
    )
    assert components.llm.api.temperature == 0.5
    store = components.vector_store
    assert store.elasticsearch == ElasticsearchSettings(
        url="http://es:9200",
        index="passages_v2",
        api_key="es-key",
        text_analyzer="french",
    )
    assert store.chroma == ChromaSettings(
        url="http://chroma:8000", path="here", collection="passages"
    )
    assert store.fake == FakeStoreSettings(seed_documents=12)


def test_a_copy_keeps_a_backend_name_the_settings_would_refuse(
    fake_settings: RagSettings,
) -> None:
    # What refuses an unknown backend is the registry of openers, not this shape.
    settings = fake_settings.model_copy(update={"embedding_backend": "gpu"})

    assert component_settings_of(settings).embedder.backend == "gpu"


def test_an_endpoint_block_gives_its_endpoint() -> None:
    block = EndpointSettings(
        url=RERANK_URL, api_format="cohere", api_key="key", model="rerank-v3.5"
    )

    assert block.endpoint() == ApiEndpoint(RERANK_URL, "cohere", "key", "rerank-v3.5")


def test_a_key_the_shape_does_not_have_is_refused() -> None:
    with pytest.raises(ValidationError, match="thread"):
        InProcessModelSettings.model_validate(
            {"batch_size": 1, "threads": 0, "thread": 2}
        )
