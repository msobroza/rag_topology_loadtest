"""The registered stage types and edge condition of the RAG, and what each declares."""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

import pytest
from pydantic import ValidationError

from query_processing.adapters.llm.base_llm import BaseLLM
from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.adapters.vector_store.base_vector_store import (
    BaseVectorStore,
    required_lexical_search,
)
from query_processing.application.edge_conditions.edge_condition import EDGE_CONDITIONS
from query_processing.application.edge_conditions.is_query_mode import is_query_mode
from query_processing.application.rag_dependencies import RagDependencies
from query_processing.application.stages.base_stage import (
    STAGE_TYPES,
    StageFields,
    StageSettings,
)
from query_processing.application.stages.generate_stage import (
    GenerateSettings,
    GenerateStage,
)
from query_processing.application.stages.retrieve_stage import (
    RetrieveSettings,
    RetrieveStage,
)
from query_processing.application.workflow_builder import open_stage
from query_processing.application.workflow_state import WorkflowState
from query_processing.domain.passages import ScoredPassage
from rag_graphs import rag_graph
from workflow_fakes import DenseOnlyStore, ScriptedStore, scored

STATE_FIELDS = set(get_type_hints(WorkflowState))
RAG_STAGE_TYPES = {
    "retrieve": (RetrieveStage, BaseVectorStore),
    "generate": (GenerateStage, BaseLLM),
}
# The user's message and the self-contained request made of it differ here, so
# each stage shows which one it reads.
ASKED = {"conversational_query": "and in May?", "standalone_query": "vacation in May"}
GENERATE_FIELDS = StageFields(("conversational_query", "reranked"), ("answer",))
TEMPLATE = {"system_prompt": "s", "user_prompt": "{question}|{context}"}


@pytest.mark.parametrize("name", sorted(RAG_STAGE_TYPES))
def test_each_rag_stage_type_is_registered_under_its_name_and_calls_a_port(
    name: str,
) -> None:
    stage_type, port = RAG_STAGE_TYPES[name]

    assert STAGE_TYPES.get(name) is stage_type
    assert stage_type.__name__ == f"{name.capitalize()}Stage"
    assert stage_type.component_type is port
    assert issubclass(stage_type.settings_type, StageSettings)


def test_embed_and_rerank_are_plain_calls_not_stage_types() -> None:
    assert sorted(STAGE_TYPES.names()) == ["generate", "retrieve"]


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
async def test_each_rag_stage_reads_and_writes_fields_of_the_workflow_state(
    fake_deps: RagDependencies, mode: str
) -> None:
    graph = rag_graph(mode)

    for name, description in graph.stages.items():
        open_stage(name, description, fake_deps.components(), STAGE_TYPES)
        assert description.reads, name
        assert description.writes, name
        assert {*description.reads, *description.writes} <= STATE_FIELDS, name


@pytest.mark.parametrize(
    ("mode", "reads", "types"),
    [
        ("dense", ("query_embedding",), [list[float]]),
        ("lexical", ("standalone_query",), [str]),
        ("hybrid", ("query_embedding", "standalone_query"), [list[float], str]),
    ],
)
def test_the_retrieve_stage_reads_what_its_retrieval_mode_needs(
    mode: Any, reads: tuple[str, ...], types: list[Any]
) -> None:
    settings = RetrieveSettings(retrieval_mode=mode, top_k=5, rrf_k=60)

    # What the start-up checks ask, with no store and no stage built.
    field_types = RetrieveStage.check_fields(settings, StageFields(reads, ("found",)))

    assert field_types.reads == dict(zip(reads, types, strict=True))
    assert field_types.writes == {"found": list[ScoredPassage]}


@pytest.mark.parametrize(
    ("mode", "fields", "said"),
    [
        (
            "dense",
            StageFields(("query_embedding", "standalone_query"), ("candidates",)),
            "the retrieve stage in retrieval_mode 'dense' reads 1 field, for "
            "embedding: list[float] "
            "(got reads ['query_embedding', 'standalone_query'])",
        ),
        (
            "lexical",
            StageFields(("standalone_query",), ()),
            "the retrieve stage in retrieval_mode 'lexical' writes 1 field, for "
            "candidates: "
            "list[ScoredPassage] (got writes [])",
        ),
    ],
)
def test_the_retrieve_stage_refuses_lists_its_retrieval_mode_cannot_use(
    mode: Any, fields: StageFields, said: str
) -> None:
    settings = RetrieveSettings(retrieval_mode=mode, top_k=5, rrf_k=60)

    with pytest.raises(ValueError, match="1 field, for") as refused:
        RetrieveStage.check_fields(settings, fields)

    assert str(refused.value) == said


def test_the_generate_stage_reads_the_question_and_the_passages() -> None:
    settings = GenerateSettings(**TEMPLATE)

    field_types = GenerateStage.check_fields(settings, GENERATE_FIELDS)

    assert field_types.reads == {
        "conversational_query": str,
        "reranked": list[ScoredPassage],
    }
    assert field_types.writes == {"answer": str}
    with pytest.raises(ValueError, match=r"reads 2 fields, for question: str, "):
        GenerateStage.check_fields(settings, StageFields(("reranked",), ("answer",)))


def test_the_fusion_constant_is_a_default_and_the_use_cases_choices_are_not() -> None:
    settings = RetrieveSettings.model_validate({"retrieval_mode": "hybrid", "top_k": 5})

    assert settings.rrf_k == 60
    with pytest.raises(ValidationError) as missing:
        RetrieveSettings.model_validate({})
    assert {error["loc"] for error in missing.value.errors()} == {
        ("retrieval_mode",),
        ("top_k",),
    }


@pytest.mark.parametrize("settings_type", [RetrieveSettings, GenerateSettings])
def test_a_stage_type_refuses_a_setting_it_does_not_know(
    settings_type: type[StageSettings],
) -> None:
    with pytest.raises(ValidationError, match="top_kk"):
        settings_type.model_validate({"top_kk": 3})


async def test_the_retrieve_stage_reads_the_fields_its_lists_name() -> None:
    store = ScriptedStore([], ["a"])
    lexical = RetrieveSettings(retrieval_mode="lexical", top_k=3, rrf_k=60)
    fields = StageFields(("rewritten",), ("found",))

    update = await RetrieveStage(store, lexical, fields).run(
        {**ASKED, "rewritten": "days off in May"}
    )

    assert store.calls == [("lexical", "days off in May", 3)]
    assert [passage.id for passage in update["found"]] == ["a"]


async def test_the_retrieve_stage_hands_a_hybrid_search_its_vector_and_words() -> None:
    store = ScriptedStore(["b", "a"], ["a"])
    hybrid = RetrieveSettings(retrieval_mode="hybrid", top_k=2, rrf_k=60)
    fields = StageFields(("query_embedding", "standalone_query"), ("candidates",))

    update = await RetrieveStage(store, hybrid, fields).run(
        {**ASKED, "query_embedding": [1.0, 0.0]}
    )

    assert sorted(call[0] for call in store.calls) == ["dense", "lexical"]
    assert ("lexical", "vacation in May", 2) in store.calls
    assert update["candidates"][0].id == "a"  # in both rankings, so fused first


async def test_the_generate_stage_answers_what_the_user_asked() -> None:
    settings = GenerateSettings(**TEMPLATE)

    update = await GenerateStage(FakeLLM(), settings, GENERATE_FIELDS).run(
        {**ASKED, "reranked": [scored("a", "May has 2 days off.")]}
    )

    assert update["answer"] == "[fake-llm] and in May?|[1] May has 2 days off."


async def test_the_generate_stage_reads_and_writes_the_fields_its_lists_name() -> None:
    settings = GenerateSettings(**TEMPLATE)
    fields = StageFields(("standalone_query", "kept"), ("reply",))

    update = await GenerateStage(FakeLLM(), settings, fields).run(
        {**ASKED, "kept": [scored("a", "2 days.")]}
    )

    assert update == {"reply": "[fake-llm] vacation in May|[1] 2 days."}


def test_the_generate_stages_settings_refuse_a_template_it_cannot_fill() -> None:
    # In the settings, so the start-up checks refuse it before any component opens.
    with pytest.raises(ValidationError, match=r"lacks the \{context\} placeholder"):
        GenerateSettings(system_prompt="s", user_prompt="Q: {question}")


def test_required_lexical_search_hands_back_a_store_that_ranks_text() -> None:
    store = ScriptedStore([], [])

    assert required_lexical_search(store, "hybrid") is store
    backend = "QP_COMPONENTS__VECTOR_STORE__BACKEND"
    with pytest.raises(ValueError, match=rf"to elasticsearch \({backend} for the one"):
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
