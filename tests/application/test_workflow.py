"""The RAG graph, built from the settings' description over fake ports, and the prompts."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

import pytest

from rag_load_test.adapters.llm.fake.fake_llm import FakeLLM
from rag_load_test.adapters.reranker.none.no_reranker import NoReranker
from rag_load_test.application.rag_dependencies import RagDependencies
from rag_load_test.application.workflow import CompiledWorkflow, stage_timings
from rag_load_test.application.workflow_builder import build_workflow
from rag_load_test.application.workflow_state import initial_state
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.corpus import synthetic_questions
from rag_load_test.domain.prompts import RagPrompts
from workflow_fakes import (
    ChainNameRecorder,
    DenseOnlyStore,
    RecordingEmbedder,
    RecordingReranker,
    ScriptedStore,
    scored,
)

VACATION_QUESTION = synthetic_questions(1)[0]
# The four stages in the graph's order, then the total: the keys of today's response.
TIMING_KEYS = ["embed", "retrieve", "rerank", "generate", "total"]
DEFAULT_TOP_K_RETRIEVE, DEFAULT_TOP_K_RERANK = 20, 10
FAKE_BACKENDS = {
    "embedding_backend": "fake",
    "rerank_backend": "fake",
    "generation_backend": "fake",
    "vector_store_backend": "fake",
}
WorkflowFor = Callable[..., CompiledWorkflow]


@pytest.fixture
def workflow_for(
    settings_factory: Callable[..., RagSettings],
) -> WorkflowFor:
    """The RAG graph of all-fake settings with ``overrides``, over ``deps``."""

    def make(deps: RagDependencies, **overrides: object) -> CompiledWorkflow:
        graph = settings_factory(**FAKE_BACKENDS, **overrides).workflow_graph()
        return build_workflow(graph, deps.components())

    return make


def scripted_deps(
    fake_deps: RagDependencies, store: ScriptedStore, embedder: RecordingEmbedder
) -> RagDependencies:
    return dataclasses.replace(
        fake_deps, vector_store=store, embedder=embedder, reranker=NoReranker()
    )


# --- the graph ----------------------------------------------------------------


async def test_query_mode_answers_from_reranked_passages(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    result = await workflow_for(fake_deps).run(VACATION_QUESTION)

    assert result.question == VACATION_QUESTION
    assert result.mode == "query"
    assert result.answer is not None
    assert result.answer.startswith("[fake-llm]")
    assert len(result.passages) == DEFAULT_TOP_K_RERANK
    scores = [p.rerank_score for p in result.passages]
    assert all(score is not None for score in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


async def test_query_mode_times_every_stage(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    result = await workflow_for(fake_deps).run(VACATION_QUESTION)

    assert list(result.timings_ms) == TIMING_KEYS
    assert result.timings_ms["total"] > 0
    assert all(ms >= 0 for ms in result.timings_ms.values())


async def test_vacation_question_retrieves_vacation_policy(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    result = await workflow_for(fake_deps).run(VACATION_QUESTION)

    topics = [p.metadata["topic"] for p in result.passages]
    # One seeded document in ten is a vacation policy; they outrank every other topic.
    assert topics[:3] == ["vacation-policy"] * 3


async def test_retrieve_mode_skips_generation(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    result = await workflow_for(fake_deps).run(VACATION_QUESTION, mode="retrieve")

    assert isinstance(fake_deps.llm, FakeLLM)
    assert fake_deps.llm.calls == 0
    assert result.mode == "retrieve"
    assert result.answer is None
    assert list(result.timings_ms) == TIMING_KEYS
    assert result.timings_ms["generate"] == 0.0
    assert result.timings_ms["total"] > 0
    assert len(result.passages) == DEFAULT_TOP_K_RERANK


async def test_no_reranker_keeps_retrieval_order(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    deps = dataclasses.replace(fake_deps, reranker=NoReranker())
    result = await workflow_for(deps).run(VACATION_QUESTION, mode="retrieve")

    vector = await deps.embedder.embed_query(VACATION_QUESTION)
    expected = await deps.vector_store.query(vector, DEFAULT_TOP_K_RETRIEVE)
    first_kept = [p.id for p in expected[:DEFAULT_TOP_K_RERANK]]
    assert [p.id for p in result.passages] == first_kept
    assert all(p.rerank_score is None for p in result.passages)


async def test_non_default_settings_size_retrieval_rerank_and_prompt(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    deps = dataclasses.replace(fake_deps, reranker=NoReranker())
    workflow = workflow_for(
        deps,
        top_k_retrieve=3,
        top_k_rerank=2,
        system_prompt="s",
        user_prompt="ASK {question} USING {context}",
    )

    result = await workflow.run("vacation days")

    assert len(result.passages) == 2
    assert result.answer is not None
    # FakeLLM echoes the first 200 characters of the user message.
    assert result.answer.startswith("[fake-llm] ASK vacation days USING [1] ")


async def test_the_reranker_gets_the_question_the_retrieve_size_and_its_own(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    reranker = RecordingReranker()
    deps = dataclasses.replace(fake_deps, reranker=reranker)
    workflow = workflow_for(deps, top_k_retrieve=7, top_k_rerank=4)

    await workflow.run(VACATION_QUESTION, mode="retrieve")

    assert reranker.calls == [(VACATION_QUESTION, 7, 4)]


def test_the_components_are_the_four_ports_by_their_fixed_names(
    fake_deps: RagDependencies,
) -> None:
    assert fake_deps.components() == {
        "embedder": fake_deps.embedder,
        "reranker": fake_deps.reranker,
        "llm": fake_deps.llm,
        "vector_store": fake_deps.vector_store,
    }


@pytest.mark.parametrize("mode", ["query", "retrieve"])
async def test_every_step_of_the_graph_reaches_a_tracer_with_a_name(
    fake_deps: RagDependencies, workflow_for: WorkflowFor, mode: Any
) -> None:
    # LangGraph passes no serialized form and names a step after its function,
    # except a lambda; MLflow's LangChain autolog (Domino tracing) then failed
    # on the unnamed step: "'NoneType' object has no attribute 'get'".
    recorder = ChainNameRecorder()

    graph = workflow_for(fake_deps).graph
    await graph.ainvoke(
        initial_state(VACATION_QUESTION, mode), {"callbacks": [recorder]}
    )

    assert None not in recorder.names, recorder.names
    assert "is_query_mode" in recorder.names  # the edge condition keeps its name


# --- retrieval modes -------------------------------------------------------------


async def test_dense_mode_asks_the_store_by_vector_only(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    store, embedder = ScriptedStore(["a", "b"], ["z"]), RecordingEmbedder()
    deps = scripted_deps(fake_deps, store, embedder)
    workflow = workflow_for(deps, top_k_retrieve=2, top_k_rerank=2)

    result = await workflow.run("vacation days", mode="retrieve")

    assert [p.id for p in result.passages] == ["a", "b"]
    assert store.calls == [("dense", [1.0, 0.0], 2)]


async def test_lexical_mode_skips_the_embed_stage_and_asks_by_words(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    store, embedder = ScriptedStore(["a"], ["z", "y"]), RecordingEmbedder()
    deps = scripted_deps(fake_deps, store, embedder)
    workflow = workflow_for(
        deps, retrieval_mode="lexical", top_k_retrieve=2, top_k_rerank=2
    )

    result = await workflow.run("vacation days", mode="retrieve")

    assert [p.id for p in result.passages] == ["z", "y"]
    assert store.calls == [("lexical", "vacation days", 2)]
    assert embedder.queries == []  # no vector is needed, so none is computed
    assert list(result.timings_ms) == ["retrieve", "rerank", "generate", "total"]
    assert result.timings_ms["retrieve"] > 0.0


async def test_hybrid_mode_asks_both_ways_and_fuses_by_rank(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    store, embedder = ScriptedStore(["a", "b", "c"], ["b", "d"]), RecordingEmbedder()
    deps = scripted_deps(fake_deps, store, embedder)
    workflow = workflow_for(
        deps, retrieval_mode="hybrid", top_k_retrieve=3, top_k_rerank=3
    )

    result = await workflow.run("vacation days", mode="retrieve")

    # b is in both rankings; a and d each lead one; c is cut by top_k_retrieve.
    assert [p.id for p in result.passages] == ["b", "a", "d"]
    assert sorted(store.calls, key=lambda call: call[0]) == [
        ("dense", [1.0, 0.0], 3),
        ("lexical", "vacation days", 3),
    ]


async def test_the_embed_stage_embeds_the_question_as_a_query(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    store, embedder = ScriptedStore(["a"], ["a"]), RecordingEmbedder()
    workflow = workflow_for(
        scripted_deps(fake_deps, store, embedder), retrieval_mode="hybrid"
    )

    result = await workflow.run("vacation days", mode="retrieve")

    # As the user asked it: the embedder puts its own prefix in front.
    assert (embedder.queries, embedder.passages) == (["vacation days"], [])
    lexical_calls = [call for call in store.calls if call[0] == "lexical"]
    assert lexical_calls == [("lexical", "vacation days", 20)]  # words, unprefixed
    assert result.question == "vacation days"


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
async def test_every_retrieval_mode_answers_over_the_fake_store(
    fake_deps: RagDependencies, workflow_for: WorkflowFor, mode: str
) -> None:
    workflow = workflow_for(fake_deps, retrieval_mode=mode)

    result = await workflow.run("How many vacation days do employees accrue?")

    assert len(result.passages) == DEFAULT_TOP_K_RERANK
    assert "vacation" in result.passages[0].text.lower()
    assert result.answer is not None


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_a_text_ranking_mode_over_a_store_that_ranks_no_text_is_refused_when_built(
    fake_deps: RagDependencies, workflow_for: WorkflowFor, mode: str
) -> None:
    deps = dataclasses.replace(fake_deps, vector_store=DenseOnlyStore())

    with pytest.raises(ValueError, match=f"retrieval_mode='{mode}'.*ranks no text"):
        workflow_for(deps, retrieval_mode=mode)


def test_dense_mode_needs_no_store_that_ranks_text(
    fake_deps: RagDependencies, workflow_for: WorkflowFor
) -> None:
    deps = dataclasses.replace(fake_deps, vector_store=DenseOnlyStore())

    workflow_for(deps, retrieval_mode="dense")  # no error


# --- the result's timings ----------------------------------------------------------


def test_stage_timings_keys_every_stage_in_the_graph_s_order_then_the_total() -> None:
    recorded = {"rerank": 3.0, "embed": 1.0}  # in the order the stages finished

    timings = stage_timings(["embed", "rerank", "generate"], recorded, 9.5)

    assert timings == {"embed": 1.0, "rerank": 3.0, "generate": 0.0, "total": 9.5}
    assert list(timings) == ["embed", "rerank", "generate", "total"]


# --- prompt helpers ----------------------------------------------------------------


def test_rag_prompts_number_the_passages_into_the_user_template() -> None:
    prompts = RagPrompts(system="Be brief.", user="C:\n{context}\nQ: {question}")
    passages = [scored("a", "Alpha text."), scored("b", "Beta text.")]

    messages = prompts.messages("What is alpha?", passages)

    assert messages == [
        {"role": "system", "content": "Be brief."},
        {
            "role": "user",
            "content": "C:\n[1] Alpha text.\n[2] Beta text.\nQ: What is alpha?",
        },
    ]


@pytest.mark.parametrize(
    ("user", "missing"),
    [("Q: {question}", "{context}"), ("Context: {context}", "{question}")],
)
def test_rag_prompts_reject_a_user_template_missing_a_placeholder(
    user: str, missing: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(missing)) as caught:
        RagPrompts(system="s", user=user)

    assert missing in str(caught.value)
    assert repr(user) in str(caught.value)


@pytest.mark.parametrize(
    "user",
    [
        "{context} {question} {history}",  # a placeholder nobody fills
        '{context} {question} answer as {"answer": ...}',  # JSON braces not doubled
        "{context} {question} }",  # a lone brace
    ],
)
def test_rag_prompts_reject_a_brace_they_cannot_fill_when_built(user: str) -> None:
    # Built at start-up from the settings, so this never waits for the first query.
    with pytest.raises(ValueError, match=r"\{\{") as caught:
        RagPrompts(system="s", user=user)

    assert "{{" in str(caught.value)  # says how to write a literal brace
    assert repr(user) in str(caught.value)


def test_rag_prompts_keep_braces_of_the_question_and_passages_as_they_are() -> None:
    prompts = RagPrompts(system="s", user='{context}|{question}|{{"k": 1}}')

    messages = prompts.messages("what is {x}?", [scored("a", "set {y} to 1")])

    assert messages[1]["content"] == '[1] set {y} to 1|what is {x}?|{"k": 1}'
