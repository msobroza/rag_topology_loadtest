"""The registered stage types and edge condition of the RAG, and what each declares."""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

import pytest
from pydantic import ValidationError

from rag_graphs import rag_graph
from rag_load_test.adapters.embedder.base_embedder import BaseEmbedder
from rag_load_test.adapters.llm.base_llm import BaseLLM
from rag_load_test.adapters.llm.fake.fake_llm import FakeLLM
from rag_load_test.adapters.reranker.base_reranker import BaseReranker
from rag_load_test.adapters.vector_store.base_vector_store import BaseVectorStore
from rag_load_test.application.edge_conditions.edge_condition import EDGE_CONDITIONS
from rag_load_test.application.edge_conditions.is_query_mode import is_query_mode
from rag_load_test.application.rag_dependencies import RagDependencies
from rag_load_test.application.stages.base_stage import STAGE_TYPES, StageSettings
from rag_load_test.application.stages.embed_stage import EmbedSettings, EmbedStage
from rag_load_test.application.stages.generate_stage import (
    GenerateSettings,
    GenerateStage,
)
from rag_load_test.application.stages.rerank_stage import RerankSettings, RerankStage
from rag_load_test.application.stages.retrieve_stage import (
    RetrieveSettings,
    RetrieveStage,
    required_lexical_search,
)
from rag_load_test.application.workflow_builder import open_stage
from rag_load_test.application.workflow_state import WorkflowState
from workflow_fakes import (
    DenseOnlyStore,
    RecordingEmbedder,
    RecordingReranker,
    ScriptedStore,
    scored,
)

STATE_FIELDS = set(get_type_hints(WorkflowState))
RAG_STAGE_TYPES = {
    "embed": (EmbedStage, BaseEmbedder),
    "retrieve": (RetrieveStage, BaseVectorStore),
    "rerank": (RerankStage, BaseReranker),
    "generate": (GenerateStage, BaseLLM),
}
# The user's message and the self-contained request made of it differ here, so
# each stage shows which one it reads.
ASKED = {"conversational_query": "and in May?", "standalone_query": "vacation in May"}


@pytest.mark.parametrize("name", sorted(RAG_STAGE_TYPES))
def test_each_rag_stage_type_is_registered_under_its_name_and_calls_a_port(
    name: str,
) -> None:
    stage_type, port = RAG_STAGE_TYPES[name]

    assert STAGE_TYPES.get(name) is stage_type
    assert stage_type.__name__ == f"{name.capitalize()}Stage"
    assert stage_type.component_type is port
    assert issubclass(stage_type.settings_type, StageSettings)


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
async def test_each_rag_stage_reads_and_writes_fields_of_the_workflow_state(
    fake_deps: RagDependencies, mode: str
) -> None:
    graph = rag_graph(mode)

    stages = [
        open_stage(name, description, fake_deps.components(), STAGE_TYPES)
        for name, description in graph.stages.items()
    ]

    for stage in stages:
        assert stage.reads, stage
        assert stage.writes, stage
        assert stage.reads | stage.writes <= STATE_FIELDS, stage


@pytest.mark.parametrize(
    ("mode", "reads"),
    [
        ("dense", {"query_embedding"}),
        ("lexical", {"standalone_query"}),
        ("hybrid", {"query_embedding", "standalone_query"}),
    ],
)
def test_the_retrieve_stage_reads_what_its_retrieval_mode_needs(
    mode: Any, reads: set[str]
) -> None:
    settings = RetrieveSettings(retrieval_mode=mode, top_k=5, rrf_k=60)

    stage = RetrieveStage(ScriptedStore([], []), settings)

    assert stage.reads == reads
    assert stage.writes == {"candidates"}
    # What the start-up checks ask, with no store and no stage built.
    assert RetrieveStage.fields_read(settings) == reads
    assert RetrieveStage.fields_written(settings) == {"candidates"}


@pytest.mark.parametrize("name", ["embed", "rerank", "generate"])
def test_a_stage_type_whose_settings_change_nothing_declares_its_class_fields(
    name: str,
) -> None:
    stage_type, _ = RAG_STAGE_TYPES[name]
    settings: Any = object()  # not read: these stage types read no setting for it

    assert stage_type.fields_read(settings) == stage_type.reads
    assert stage_type.fields_written(settings) == stage_type.writes


def test_the_fusion_constant_is_a_default_and_the_use_cases_choices_are_not() -> None:
    settings = RetrieveSettings.model_validate({"retrieval_mode": "hybrid", "top_k": 5})

    assert settings.rrf_k == 60
    with pytest.raises(ValidationError) as missing:
        RetrieveSettings.model_validate({})
    assert {error["loc"] for error in missing.value.errors()} == {
        ("retrieval_mode",),
        ("top_k",),
    }


@pytest.mark.parametrize(
    "settings_type", [EmbedSettings, RetrieveSettings, RerankSettings, GenerateSettings]
)
def test_a_stage_type_refuses_a_setting_it_does_not_know(
    settings_type: type[StageSettings],
) -> None:
    with pytest.raises(ValidationError, match="top_kk"):
        settings_type.model_validate({"top_kk": 3})


async def test_the_stages_that_search_read_the_standalone_query() -> None:
    embedder, reranker = RecordingEmbedder(), RecordingReranker()
    store = ScriptedStore([], ["a"])
    lexical = RetrieveSettings(retrieval_mode="lexical", top_k=3, rrf_k=60)

    await EmbedStage(embedder, EmbedSettings()).run({**ASKED})
    await RetrieveStage(store, lexical).run({**ASKED})
    await RerankStage(reranker, RerankSettings(top_k=1)).run(
        {**ASKED, "candidates": [scored("a")]}
    )

    assert embedder.queries == ["vacation in May"]
    assert store.calls == [("lexical", "vacation in May", 3)]
    assert reranker.calls == [("vacation in May", 1, 1)]


async def test_the_generate_stage_answers_what_the_user_asked() -> None:
    settings = GenerateSettings(system_prompt="s", user_prompt="{question}|{context}")

    update = await GenerateStage(FakeLLM(), settings).run(
        {**ASKED, "reranked": [scored("a", "May has 2 days off.")]}
    )

    assert update["answer"] == "[fake-llm] and in May?|[1] May has 2 days off."


def test_the_generate_stages_settings_refuse_a_template_it_cannot_fill() -> None:
    # In the settings, so the start-up checks refuse it before any component opens.
    with pytest.raises(ValidationError, match=r"lacks the \{context\} placeholder"):
        GenerateSettings(system_prompt="s", user_prompt="Q: {question}")


def test_required_lexical_search_hands_back_a_store_that_ranks_text() -> None:
    store = ScriptedStore([], [])

    assert required_lexical_search(store, "hybrid") is store
    with pytest.raises(ValueError, match="RAG_VECTOR_STORE_BACKEND=elasticsearch"):
        required_lexical_search(DenseOnlyStore(), "lexical")


# --- the edge condition --------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "expected"),
    [({"mode": "query"}, True), ({"mode": "retrieve"}, False), ({}, False)],
)
def test_is_query_mode_is_a_registered_question_about_the_state_alone(
    state: WorkflowState, expected: bool
) -> None:
    assert EDGE_CONDITIONS.get("is_query_mode") is is_query_mode
    assert list(inspect.signature(is_query_mode).parameters) == ["state"]
    assert is_query_mode(state) is expected
