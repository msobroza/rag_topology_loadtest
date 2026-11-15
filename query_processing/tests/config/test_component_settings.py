"""The components of the workflow file: their types, backends, aliases and blocks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_graphs import (
    packaged_components,
    packaged_graph,
    rag_graph,
    write_workflow_file,
)
from rag_load_test.config.component_settings import (
    ChromaSettings,
    ComponentHttpSettings,
    ElasticsearchSettings,
    EmbedderEndpointSettings,
    EmbedderLocalSettings,
    EndpointSettings,
    FakeStoreSettings,
    LlmEndpointSettings,
    RerankerEndpointSettings,
    RerankerLocalSettings,
    RerankerSettings,
    alias_of,
    typed_by_name,
)
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.config.workflow_file import WORKFLOW_FILE_VARIABLE
from rag_load_test.domain.api_endpoint import ApiEndpoint

SettingsFactory = Callable[..., RagSettings]
EMBED_URL = "http://embedder:8002/v3/embeddings"
RERANK_URL = "http://reranker:8001/v3/rerank"


def test_the_packaged_file_declares_the_rags_three_models_and_its_store(
    settings_factory: SettingsFactory,
) -> None:
    components = settings_factory().components

    assert list(components) == ["embedder", "reranker", "llm", "vector_store"]
    embedder = components["embedder"]
    assert (embedder.type, embedder.backend) == ("embedder", "local")
    assert embedder.source_id == "Snowflake/snowflake-arctic-embed-l-v2.0"
    assert embedder.alias == "snowflake-arctic-embed-l-v2.0"
    assert embedder.query_prefix == "query: "
    assert embedder.local == EmbedderLocalSettings(batch_size=64, threads=0)
    assert embedder.api == EmbedderEndpointSettings(
        url="", api_format="openai", api_key="", model="", batch_size=64
    )
    assert embedder.http == ComponentHttpSettings()  # every value the default
    reranker = components["reranker"]
    assert (reranker.type, reranker.backend) == ("reranker", "local")
    assert reranker.source_id == "antoinelouis/crossencoder-camembert-base-mmarcoFR"
    assert reranker.alias == "crossencoder-camembert-base-mmarcoFR"
    assert reranker.local == RerankerLocalSettings(
        batch_size=0, threads=0, calibration_passages=20
    )
    assert reranker.api == RerankerEndpointSettings(api_format="cohere")
    assert reranker.http == ComponentHttpSettings()
    llm = components["llm"]
    assert (llm.type, llm.backend) == ("llm", "api")
    assert llm.api == LlmEndpointSettings(
        url="https://api.openai.com/v1/chat/completions",
        api_format="openai",
        api_key="",
        model="gpt-4o-mini",
        temperature=0.0,
    )
    # An answer takes longer than a model-server call: its own timeout.
    assert llm.http == ComponentHttpSettings(timeout_s=120.0)


def test_every_value_is_reachable_by_its_environment_path(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    variables = {
        "RAG_COMPONENTS__EMBEDDER__BACKEND": "api",
        "RAG_COMPONENTS__EMBEDDER__SOURCE_ID": "/models/my-embedder",
        "RAG_COMPONENTS__EMBEDDER__QUERY_PREFIX": "",
        "RAG_COMPONENTS__EMBEDDER__LOCAL__BATCH_SIZE": "8",
        "RAG_COMPONENTS__EMBEDDER__LOCAL__THREADS": "3",
        "RAG_COMPONENTS__EMBEDDER__API__URL": EMBED_URL,
        "RAG_COMPONENTS__EMBEDDER__API__API_KEY": "domino",
        "RAG_COMPONENTS__EMBEDDER__API__BATCH_SIZE": "16",
        "RAG_COMPONENTS__EMBEDDER__HTTP__RETRIES": "5",
        "RAG_COMPONENTS__RERANKER__ALIAS": "rr",
        "RAG_COMPONENTS__RERANKER__LOCAL__BATCH_SIZE": "4",
        "RAG_COMPONENTS__RERANKER__LOCAL__THREADS": "2",
        "RAG_COMPONENTS__RERANKER__LOCAL__CALIBRATION_PASSAGES": "12",
        "RAG_COMPONENTS__RERANKER__HTTP__BACKOFF_S": "0.5",
        "RAG_COMPONENTS__LLM__API__MODEL": "gpt-x",
        "RAG_COMPONENTS__LLM__API__API_KEY": "sk-key",
        "RAG_COMPONENTS__LLM__API__TEMPERATURE": "0.5",
        "RAG_COMPONENTS__LLM__HTTP__TIMEOUT_S": "9",
    }
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    components = settings_factory().components

    embedder, reranker, llm = (components[n] for n in ("embedder", "reranker", "llm"))
    assert (embedder.backend, embedder.alias, embedder.query_prefix) == (
        "api",
        "my-embedder",
        "",
    )
    assert embedder.local == EmbedderLocalSettings(batch_size=8, threads=3)
    assert (embedder.api.url, embedder.api.api_key) == (EMBED_URL, "domino")
    assert embedder.api.batch_size == 16
    assert embedder.http == ComponentHttpSettings(retries=5)
    assert reranker.alias == "rr"
    assert reranker.source_id == "antoinelouis/crossencoder-camembert-base-mmarcoFR"
    assert reranker.local == RerankerLocalSettings(
        batch_size=4, threads=2, calibration_passages=12
    )
    assert reranker.http == ComponentHttpSettings(backoff_s=0.5)
    assert (llm.api.model, llm.api.api_key, llm.api.temperature) == (
        "gpt-x",
        "sk-key",
        0.5,
    )
    assert llm.api.url == "https://api.openai.com/v1/chat/completions"  # kept
    assert llm.http == ComponentHttpSettings(timeout_s=9.0)


@pytest.mark.parametrize(
    ("source_id", "alias"),
    [
        ("Snowflake/snowflake-arctic-embed-l-v2.0", "snowflake-arctic-embed-l-v2.0"),
        ("/domino/datasets/local/d/models/crossencoder", "crossencoder"),
        ("models/reranker/", "reranker"),
        ("bge-small", "bge-small"),
    ],
    ids=["hub-id", "folder", "folder-with-a-slash", "bare-name"],
)
def test_the_alias_defaults_to_the_last_part_of_the_source_id(
    source_id: str, alias: str
) -> None:
    assert alias_of(source_id) == alias
    assert RerankerSettings(backend="local", source_id=source_id).alias == alias


def test_an_alias_given_wins_over_the_source_id() -> None:
    reranker = RerankerSettings(backend="local", source_id="org/model", alias="rr")

    assert reranker.alias == "rr"


def test_a_backends_settings_sit_in_the_block_named_after_it(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(
        components__reranker__backend="api",
        components__reranker__api__url=RERANK_URL,
        components__reranker__local__threads=2,
    )

    reranker = settings.components["reranker"]
    assert (reranker.api.url, reranker.local.threads) == (RERANK_URL, 2)


@pytest.mark.parametrize(
    ("component", "block"),
    [("llm", "local"), ("embedder", "none"), ("reranker", "elasticsearch")],
)
def test_a_block_of_a_backend_the_type_does_not_have_is_refused(
    settings_factory: SettingsFactory, component: str, block: str
) -> None:
    with pytest.raises(ValidationError) as refused:
        settings_factory(components={component: {block: {"threads": 2}}})

    message = str(refused.value)
    assert f"the {component!r} component type has no {block!r}" in message
    assert "a backend's settings in the block named after that backend" in message


def test_a_key_of_a_block_it_does_not_have_is_refused() -> None:
    with pytest.raises(ValidationError, match="thread"):
        EmbedderLocalSettings.model_validate({"batch_size": 1, "thread": 2})


def test_a_component_is_of_the_type_of_its_name_unless_it_says(
    settings_factory: SettingsFactory,
) -> None:
    coarse = {"type": "reranker", "backend": "fake", "source_id": "org/coarse-rr"}

    settings = settings_factory(components={"coarse": coarse})

    assert settings.components["coarse"].type == "reranker"
    assert settings.components["coarse"].alias == "coarse-rr"
    assert settings.component_types() == {
        "embedder": "embedder",
        "reranker": "reranker",
        "llm": "llm",
        "coarse": "reranker",
        "vector_store": "vector_store",
    }


def test_a_component_of_an_unknown_type_is_refused_naming_the_types(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError) as refused:
        settings_factory(components={"encoder": {"backend": "fake", "source_id": "x"}})

    message = str(refused.value)
    assert "components.encoder" in message
    assert "'encoder'" in message
    assert "'embedder', 'reranker', 'llm'" in message


def test_typed_by_name_leaves_a_given_type_and_what_is_no_mapping() -> None:
    assert typed_by_name({"a": {"type": "llm"}, "b": "x"}) == {
        "a": {"type": "llm"},
        "b": "x",
    }
    assert typed_by_name(None) is None


def test_any_backend_name_is_kept_for_the_openers_to_refuse(
    settings_factory: SettingsFactory,
) -> None:
    # What refuses an unknown backend is the registry of openers, not this shape.
    settings = settings_factory(components__embedder__backend="gpu")

    assert settings.components["embedder"].backend == "gpu"


def test_the_calibration_request_reads_no_stages_setting(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(graph=rag_graph(retrieve_top_k=7, rerank_top_k=3))

    local = settings.components["reranker"].local
    assert local.calibration_passages == 20  # the reranker's own default


def test_an_endpoint_block_gives_its_endpoint_its_model_else_the_alias() -> None:
    block = EndpointSettings(
        url=RERANK_URL, api_format="cohere", api_key="key", model="rerank-v3.5"
    )

    assert block.endpoint("rr") == ApiEndpoint(
        RERANK_URL, "cohere", "key", "rerank-v3.5"
    )
    unnamed = block.model_copy(update={"model": ""})
    assert unnamed.endpoint("rr") == ApiEndpoint(RERANK_URL, "cohere", "key", "rr")


# --- the vector store ------------------------------------------------------------------


def test_the_vector_store_is_a_component_of_the_packaged_file(
    settings_factory: SettingsFactory,
) -> None:
    store = settings_factory().components["vector_store"]

    assert (store.type, store.backend) == ("vector_store", "elasticsearch")
    assert store.elasticsearch == ElasticsearchSettings(
        url="http://localhost:9200", index="rag_passages"
    )
    assert store.elasticsearch.text_analyzer == "standard"
    assert store.chroma == ChromaSettings(
        url="", path="chroma", collection="rag_passages"
    )
    assert store.fake == FakeStoreSettings(seed_documents=200)


def test_each_block_of_the_vector_store_is_read_by_its_path(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = "RAG_COMPONENTS__VECTOR_STORE__"
    for path, value in {
        "BACKEND": "chroma",
        "ELASTICSEARCH__URL": "http://es:9200",
        "ELASTICSEARCH__INDEX": "passages_v2",
        "ELASTICSEARCH__API_KEY": "es-key",
        "ELASTICSEARCH__TEXT_ANALYZER": "french",
        "CHROMA__URL": "http://chroma:8000",
        "CHROMA__PATH": "here",
        "CHROMA__COLLECTION": "passages",
        "FAKE__SEED_DOCUMENTS": "12",
    }.items():
        monkeypatch.setenv(prefix + path, value)

    store = settings_factory().components["vector_store"]

    assert store.backend == "chroma"
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


def test_a_block_the_vector_store_does_not_have_is_refused_naming_its_keys(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError) as refused:
        settings_factory(components__vector_store__qdrant__url="http://q:6333")

    message = str(refused.value)
    assert "the 'vector_store' component type has no 'qdrant'" in message
    assert "(backend, chroma, elasticsearch, fake, type)" in message


def test_a_vector_store_of_another_name_states_its_type(
    settings_factory: SettingsFactory,
) -> None:
    passages = {
        "type": "vector_store",
        "backend": "fake",
        "fake": {"seed_documents": 5},
    }

    settings = settings_factory(components={"passages": passages})

    assert settings.component_types()["passages"] == "vector_store"
    assert settings.components["passages"].fake == FakeStoreSettings(seed_documents=5)


def test_the_model_components_are_every_component_but_the_store(
    settings_factory: SettingsFactory,
) -> None:
    models = settings_factory().model_components()

    assert list(models) == ["embedder", "reranker", "llm"]


def test_ingestion_fills_the_one_vector_store_the_file_declares(
    settings_factory: SettingsFactory,
) -> None:
    assert settings_factory().ingestion_store() == "vector_store"
    passages = {"type": "vector_store", "backend": "fake"}

    with pytest.raises(ValueError, match=r"declares 2 .*'vector_store', 'passages'"):
        settings_factory(components={"passages": passages}).ingestion_store()


def test_ingestion_refuses_a_file_without_a_vector_store(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    components = packaged_components()
    del components["vector_store"]
    workflow = write_workflow_file(tmp_path / "w.yaml", packaged_graph(), components)
    monkeypatch.setenv(WORKFLOW_FILE_VARIABLE, str(workflow))

    with pytest.raises(ValueError, match=r"declares 0 \(its vector stores: none\)"):
        settings_factory().ingestion_store()
