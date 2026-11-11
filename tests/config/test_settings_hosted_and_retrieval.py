"""``RagSettings``: default models, hosted APIs, retrieval mode, temperature, runtime."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from rag_load_test.config.settings import RagSettings, RerankApi

SettingsFactory = Callable[..., RagSettings]
PROVIDER_VARIABLES = ("COHERE_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL")


@pytest.fixture(autouse=True)
def no_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own keys must not leak into what these tests assert."""
    for name in PROVIDER_VARIABLES:
        monkeypatch.delenv(name, raising=False)


# --- defaults ------------------------------------------------------------------


def test_default_models_and_their_served_aliases(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory()

    assert s.embedder_model == "Snowflake/snowflake-arctic-embed-l-v2.0"
    assert s.ovms_embeddings_model == "snowflake-arctic-embed-l-v2.0"
    assert s.reranker_model == "antoinelouis/crossencoder-camembert-base-mmarcoFR"
    assert s.ovms_rerank_model == "crossencoder-camembert-base-mmarcoFR"


def test_defaults_of_the_new_options(settings_factory: SettingsFactory) -> None:
    s = settings_factory()

    assert s.llm_temperature == 0.0
    assert (s.retrieval_mode, s.hybrid_rrf_k) == ("dense", 60)
    assert (s.bm25_k1, s.bm25_b) == (1.2, 0.75)
    assert (s.local_model_runtime, s.openvino_models_dir) == (
        "openvino",
        "models/openvino",
    )
    assert (s.chroma_url, s.chroma_path, s.chroma_collection) == (
        "",
        "chroma",
        "rag_passages",
    )
    assert s.openai_embedding_model == "text-embedding-3-small"
    assert (s.cohere_api_key, s.openai_api_key, s.openai_base_url) == ("", "", "")


# --- the environment -----------------------------------------------------------


def test_the_new_options_read_rag_prefixed_variables(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("RAG_VECTOR_STORE_BACKEND", "chroma")
    monkeypatch.setenv("RAG_LOCAL_MODEL_RUNTIME", "torch")

    s = settings_factory()

    assert s.llm_temperature == 0.7
    assert (s.retrieval_mode, s.vector_store_backend) == ("hybrid", "chroma")
    assert s.local_model_runtime == "torch"


def test_provider_keys_are_read_without_the_rag_prefix(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "co-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("RAG_COHERE_API_KEY", "must-not-be-read")

    s = settings_factory()

    assert (s.cohere_api_key, s.openai_api_key) == ("co-key", "sk-key")
    assert s.openai_base_url == "https://gateway.example/v1"


# --- validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_temperature", -0.1),
        ("llm_temperature", 2.1),
        ("hybrid_rrf_k", 0),
        ("bm25_k1", -1.0),
        ("bm25_b", 1.5),
        ("retrieval_mode", "sparse"),
        ("local_model_runtime", "onnx"),
        ("vector_store_backend", "qdrant"),
    ],
)
def test_an_out_of_range_option_is_refused_at_construction(
    settings_factory: SettingsFactory, field: str, value: object
) -> None:
    with pytest.raises(ValidationError, match=field):
        settings_factory(**{field: value})


@pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
def test_the_temperature_may_be_anything_the_chat_api_accepts(
    settings_factory: SettingsFactory, temperature: float
) -> None:
    assert settings_factory(llm_temperature=temperature).llm_temperature == temperature


# --- hosted backends ---------------------------------------------------------------


@pytest.mark.parametrize("topology", ["monolith", "split-reranker", "split-all"])
def test_hosted_backends_fit_any_topology_and_name_the_run(
    settings_factory: SettingsFactory, topology: str
) -> None:
    s = settings_factory(
        topology=topology, embedder_backend="openai", reranker_backend="cohere"
    )

    assert (s.embedder_backend, s.reranker_backend) == ("openai", "cohere")
    assert s.run_label == f"{topology}+embedder=openai+reranker=cohere"


def test_the_cohere_reranker_points_at_cohere_with_its_key(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory(reranker_backend="cohere", cohere_api_key="co-key")

    assert s.rerank_api() == RerankApi(
        "https://api.cohere.com/v2/rerank", "rerank-v3.5", "co-key"
    )


def test_the_cohere_route_and_model_can_point_at_a_compatible_server(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory(
        reranker_backend="cohere",
        cohere_rerank_url="http://localhost:7997/rerank",
        cohere_rerank_model="bge-reranker-v2-m3",
    )

    assert s.rerank_api() == RerankApi(
        "http://localhost:7997/rerank", "bge-reranker-v2-m3", ""
    )


def test_the_openai_reranker_posts_to_the_rerank_route_of_the_base_url(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory(
        reranker_backend="openai",
        openai_base_url="https://gateway.example/v1/",
        openai_rerank_model="rerank-model",
        openai_api_key="sk-key",
    )

    assert s.rerank_api() == RerankApi(
        "https://gateway.example/v1/rerank", "rerank-model", "sk-key"
    )


@pytest.mark.parametrize(
    "given",
    [
        {},
        {"openai_base_url": "https://gateway.example/v1"},
        {"openai_rerank_model": "rerank-model"},
    ],
)
def test_the_openai_reranker_needs_a_base_url_and_a_model(
    settings_factory: SettingsFactory, given: dict[str, str]
) -> None:
    with pytest.raises(ValidationError) as caught:
        settings_factory(reranker_backend="openai", **given)

    message = str(caught.value)
    assert "RAG_RERANKER_BACKEND='openai'" in message
    assert "OPENAI_BASE_URL" in message
    assert "RAG_OPENAI_RERANK_MODEL" in message


def test_rerank_api_refuses_a_backend_that_is_not_a_hosted_api(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(ValueError, match="RAG_RERANKER_BACKEND='local'"):
        settings_factory().rerank_api()


# --- retrieval -------------------------------------------------------------------


def test_workflow_parameters_carry_the_retrieval_mode_and_the_fusion_constant(
    settings_factory: SettingsFactory,
) -> None:
    parameters = settings_factory(
        retrieval_mode="hybrid", hybrid_rrf_k=10
    ).workflow_parameters()

    assert (parameters.retrieval_mode, parameters.rrf_k) == ("hybrid", 10)


def test_the_retrieval_mode_and_the_store_name_the_run(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory(vector_store_backend="chroma", retrieval_mode="lexical")

    assert s.run_label == "monolith+store=chroma+retrieval=lexical"


@pytest.mark.parametrize(
    ("overrides", "prefix"),
    [
        ({}, "query: "),  # monolith: our model in-process
        ({"topology": "split-all"}, "query: "),  # our model behind a model server
        ({"embedder_backend": "openai"}, ""),
        ({"embedder_backend": "fake"}, ""),
        ({"embedder_query_prefix": ""}, ""),
    ],
)
def test_the_query_prefix_goes_only_to_an_embedder_serving_our_model(
    settings_factory: SettingsFactory, overrides: dict[str, str], prefix: str
) -> None:
    assert settings_factory(**overrides).workflow_parameters().query_prefix == prefix
