"""``WorkflowSettings``: default models, temperature, runtime, store, retrieval."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from query_processing.config.workflow_settings import WorkflowSettings
from rag_graphs import rag_graph

SettingsFactory = Callable[..., WorkflowSettings]
TWO_MODELS = ("embedder", "reranker")


# --- defaults ------------------------------------------------------------------


def test_the_files_models_and_their_served_aliases(
    settings_factory: SettingsFactory,
) -> None:
    embedder, reranker = (settings_factory().components[n] for n in TWO_MODELS)

    assert embedder.source_id == "fixture-org/fixture-embedder"
    assert embedder.alias == "fixture-embedder"
    assert reranker.source_id == "fixture-org/fixture-reranker"
    assert reranker.alias == "fixture-reranker"


def test_defaults_of_the_new_options(settings_factory: SettingsFactory) -> None:
    s = settings_factory()

    assert s.components["llm"].api.temperature == 0.0
    assert (s.local_models.runtime, s.local_models.openvino_dir) == (
        "openvino",
        "models/openvino",
    )
    chroma = s.components["vector_store"].chroma
    assert (chroma.url, chroma.path, chroma.collection) == (
        "",
        "chroma",
        "fixture_passages",  # the file's: a collection has no default
    )


# --- the environment -----------------------------------------------------------


def test_the_new_options_read_qp_prefixed_variables(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QP_COMPONENTS__LLM__API__TEMPERATURE", "0.7")
    monkeypatch.setenv("QP_COMPONENTS__VECTOR_STORE__BACKEND", "fake")
    monkeypatch.setenv("QP_LOCAL_MODELS__RUNTIME", "torch")

    s = settings_factory()

    assert s.components["llm"].api.temperature == 0.7
    assert s.components["vector_store"].backend == "fake"
    assert s.local_models.runtime == "torch"


# --- validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("components__llm__api__temperature", -0.1),
        ("components__llm__api__temperature", 2.1),
        ("local_models__runtime", "onnx"),
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
    s = settings_factory(
        graph=rag_graph(mode), components__vector_store__backend="chroma"
    )

    assert s.graph.stages["retrieve"].stage_settings["retrieval_mode"] == mode


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_the_default_run_label_names_the_components_not_the_retrieval_mode(
    settings_factory: SettingsFactory, mode: str
) -> None:
    # Each component with its backend; a retrieval variant sets its own label.
    s = settings_factory(
        components__vector_store__backend="fake", graph=rag_graph(mode)
    )

    assert s.run_label == "embedder=local+reranker=local+llm=api+vector_store=fake"
