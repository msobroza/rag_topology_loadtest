"""``RagSettings``: default models, retrieval mode, temperature, runtime, store."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from rag_load_test.config.settings import RagSettings

SettingsFactory = Callable[..., RagSettings]
EMBED_URL = "http://embedder/v3/embeddings"
SPLIT_ALL = {
    "topology": "split-all",
    "embedding_api_url": EMBED_URL,
    "rerank_api_url": "http://reranker/v3/rerank",
}


# --- defaults ------------------------------------------------------------------


def test_default_models_and_their_served_aliases(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory()

    assert s.embedding_model == "Snowflake/snowflake-arctic-embed-l-v2.0"
    assert s.embedding_api_model == "snowflake-arctic-embed-l-v2.0"
    assert s.rerank_model == "antoinelouis/crossencoder-camembert-base-mmarcoFR"
    assert s.rerank_api_model == "crossencoder-camembert-base-mmarcoFR"


def test_defaults_of_the_new_options(settings_factory: SettingsFactory) -> None:
    s = settings_factory()

    assert s.generation_temperature == 0.0
    assert (s.retrieval_mode, s.hybrid_rrf_k) == ("dense", 60)
    assert (s.local_model_runtime, s.openvino_models_dir) == (
        "openvino",
        "models/openvino",
    )
    assert (s.chroma_url, s.chroma_path, s.chroma_collection) == (
        "",
        "chroma",
        "rag_passages",
    )


# --- the environment -----------------------------------------------------------


def test_the_new_options_read_rag_prefixed_variables(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_GENERATION_TEMPERATURE", "0.7")
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("RAG_VECTOR_STORE_BACKEND", "fake")
    monkeypatch.setenv("RAG_LOCAL_MODEL_RUNTIME", "torch")

    s = settings_factory()

    assert s.generation_temperature == 0.7
    assert (s.retrieval_mode, s.vector_store_backend) == ("hybrid", "fake")
    assert s.local_model_runtime == "torch"


# --- validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation_temperature", -0.1),
        ("generation_temperature", 2.1),
        ("hybrid_rrf_k", 0),
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


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_a_text_ranking_mode_on_chroma_is_refused_with_the_values_and_the_fix(
    settings_factory: SettingsFactory, mode: str
) -> None:
    with pytest.raises(ValidationError) as info:
        settings_factory(retrieval_mode=mode, vector_store_backend="chroma")

    message = str(info.value)
    assert f"RAG_RETRIEVAL_MODE={mode!r} needs a store that ranks text" in message
    assert "RAG_VECTOR_STORE_BACKEND='chroma' has none" in message
    assert "use 'elasticsearch' or 'fake', or RAG_RETRIEVAL_MODE=dense" in message


@pytest.mark.parametrize(
    ("mode", "store"),
    [("dense", "chroma"), ("hybrid", "elasticsearch"), ("lexical", "fake")],
)
def test_a_retrieval_mode_its_store_can_serve_is_accepted(
    settings_factory: SettingsFactory, mode: str, store: str
) -> None:
    s = settings_factory(retrieval_mode=mode, vector_store_backend=store)

    assert (s.retrieval_mode, s.vector_store_backend) == (mode, store)


@pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
def test_the_temperature_may_be_anything_the_chat_api_accepts(
    settings_factory: SettingsFactory, temperature: float
) -> None:
    assert (
        settings_factory(generation_temperature=temperature).generation_temperature
        == temperature
    )


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
    s = settings_factory(vector_store_backend="fake", retrieval_mode="lexical")

    assert s.run_label == "monolith+store=fake+retrieval=lexical"


@pytest.mark.parametrize(
    ("overrides", "prefix"),
    [
        ({}, "query: "),  # monolith: the model in-process
        (SPLIT_ALL, "query: "),  # our model behind a model server
        ({"embedding_backend": "api", "embedding_api_url": EMBED_URL}, "query: "),
        ({"embedding_backend": "fake"}, ""),
        ({"embedding_query_prefix": ""}, ""),
    ],
)
def test_the_query_prefix_goes_to_every_embedder_but_the_fake(
    settings_factory: SettingsFactory, overrides: dict[str, str], prefix: str
) -> None:
    # The fake embedder runs no model, so it has no query convention.
    assert settings_factory(**overrides).workflow_parameters().query_prefix == prefix
