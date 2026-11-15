"""``RagSettings``: default models, temperature, runtime, store, the retrieval label."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from rag_graphs import rag_graph
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.graph_description import GraphDescription

SettingsFactory = Callable[..., RagSettings]
TWO_MODELS = ("embedder", "reranker")


# --- defaults ------------------------------------------------------------------


def test_default_models_and_their_served_aliases(
    settings_factory: SettingsFactory,
) -> None:
    embedder, reranker = (settings_factory().components[n] for n in TWO_MODELS)

    assert embedder.source_id == "Snowflake/snowflake-arctic-embed-l-v2.0"
    assert embedder.alias == "snowflake-arctic-embed-l-v2.0"
    assert reranker.source_id == "antoinelouis/crossencoder-camembert-base-mmarcoFR"
    assert reranker.alias == "crossencoder-camembert-base-mmarcoFR"


def test_defaults_of_the_new_options(settings_factory: SettingsFactory) -> None:
    s = settings_factory()

    assert s.components["llm"].api.temperature == 0.0
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
    monkeypatch.setenv("RAG_COMPONENTS__LLM__API__TEMPERATURE", "0.7")
    monkeypatch.setenv("RAG_VECTOR_STORE_BACKEND", "fake")
    monkeypatch.setenv("RAG_LOCAL_MODEL_RUNTIME", "torch")

    s = settings_factory()

    assert s.components["llm"].api.temperature == 0.7
    assert s.vector_store_backend == "fake"
    assert s.local_model_runtime == "torch"


# --- validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("components__llm__api__temperature", -0.1),
        ("components__llm__api__temperature", 2.1),
        ("local_model_runtime", "onnx"),
        ("vector_store_backend", "qdrant"),
    ],
)
def test_an_out_of_range_option_is_refused_at_construction(
    settings_factory: SettingsFactory, field: str, value: object
) -> None:
    with pytest.raises(ValidationError, match=field.split("__")[-1]):
        settings_factory(**{field: value})


@pytest.mark.parametrize("temperature", [0.0, 1.0, 2.0])
def test_the_temperature_may_be_anything_the_generation_api_accepts(
    settings_factory: SettingsFactory, temperature: float
) -> None:
    settings = settings_factory(components__llm__api__temperature=temperature)

    assert settings.components["llm"].api.temperature == temperature


# --- retrieval -------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_a_text_ranking_mode_on_chroma_is_left_to_the_retrieve_stage(
    settings_factory: SettingsFactory, mode: str
) -> None:
    # One refusal only, the retrieve stage's when the graph is built at start-up.
    s = settings_factory(graph=rag_graph(mode), vector_store_backend="chroma")

    assert s.graph.stages["retrieve"].stage_settings["retrieval_mode"] == mode


def test_the_retrieval_mode_and_the_store_name_the_run(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory(vector_store_backend="fake", graph=rag_graph("lexical"))

    assert s.run_label == "monolith+store=fake+retrieval=lexical"


def test_the_run_label_reads_the_retrieval_mode_from_the_graph(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_GRAPH__STAGES__RETRIEVE__RETRIEVAL_MODE", "hybrid")

    assert settings_factory().run_label == "monolith+retrieval=hybrid"


def test_a_graph_without_a_retrieve_stage_names_no_retrieval_mode(
    settings_factory: SettingsFactory,
) -> None:
    graph = GraphDescription.model_validate(
        {"stages": {"rerank": {"call": "reranker", "top_k": 1}}}
    )

    assert settings_factory(graph=graph).run_label == "monolith"
