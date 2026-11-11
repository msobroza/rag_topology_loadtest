"""Workflow tests: the graph over fake ports and the prompt helpers."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence

import pytest

from rag_load_test.adapters.fakes.fake_models import FakeChatModel
from rag_load_test.adapters.local_models.sentence_transformer_models import NoReranker
from rag_load_test.application.workflow import RagWorkflow, merge_timings
from rag_load_test.config.settings import RagSettings
from rag_load_test.domain.corpus import synthetic_questions
from rag_load_test.domain.passages import STAGES, Passage, ScoredPassage
from rag_load_test.domain.ports import (
    EmbedderPort,
    RagDependencies,
    RerankerPort,
    VectorStorePort,
)
from rag_load_test.domain.prompts import RagPrompts
from rag_load_test.domain.workflow_parameters import WorkflowParameters

VACATION_QUESTION = synthetic_questions(1)[0]
TIMING_KEYS = {*STAGES, "total"}
DEFAULT_TOP_K_RETRIEVE, DEFAULT_TOP_K_RERANK = 20, 5


def workflow_parameters(**overrides: object) -> WorkflowParameters:
    """Dense retrieval of 20, 5 kept, no query prefix; ``overrides`` win."""
    values: dict[str, object] = {
        "top_k_retrieve": 20,
        "top_k_rerank": 5,
        "prompts": RagPrompts(system="s", user="{question} {context}"),
        "retrieval_mode": "dense",
        "rrf_k": 60,
        "query_prefix": "",
    }
    return WorkflowParameters(**{**values, **overrides})  # type: ignore[arg-type]


@pytest.fixture
def parameters(fake_settings: RagSettings) -> WorkflowParameters:
    """The default sizes and the packaged prompts."""
    return fake_settings.workflow_parameters()


# --- the graph ----------------------------------------------------------------


async def test_query_mode_answers_from_reranked_passages(
    fake_deps: RagDependencies, parameters: WorkflowParameters
) -> None:
    result = await RagWorkflow(fake_deps, parameters).run(VACATION_QUESTION)

    assert result.question == VACATION_QUESTION
    assert result.mode == "query"
    assert result.answer is not None
    assert result.answer.startswith("[fake-llm]")
    assert len(result.passages) == DEFAULT_TOP_K_RERANK
    scores = [p.rerank_score for p in result.passages]
    assert all(score is not None for score in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


async def test_query_mode_times_every_stage(
    fake_deps: RagDependencies, parameters: WorkflowParameters
) -> None:
    result = await RagWorkflow(fake_deps, parameters).run(VACATION_QUESTION)

    assert set(result.timings_ms) == TIMING_KEYS
    assert result.timings_ms["total"] > 0
    assert all(ms >= 0 for ms in result.timings_ms.values())


async def test_vacation_question_retrieves_vacation_policy(
    fake_deps: RagDependencies, parameters: WorkflowParameters
) -> None:
    result = await RagWorkflow(fake_deps, parameters).run(VACATION_QUESTION)

    topics = [p.metadata["topic"] for p in result.passages]
    # One seeded document in ten is a vacation policy; they outrank every other topic.
    assert topics[:3] == ["vacation-policy"] * 3


async def test_retrieve_mode_skips_generation(
    fake_deps: RagDependencies, parameters: WorkflowParameters
) -> None:
    workflow = RagWorkflow(fake_deps, parameters)
    result = await workflow.run(VACATION_QUESTION, mode="retrieve")

    assert isinstance(fake_deps.chat_model, FakeChatModel)
    assert fake_deps.chat_model.calls == 0
    assert result.mode == "retrieve"
    assert result.answer is None
    assert result.timings_ms["generate"] == 0.0
    assert result.timings_ms["total"] > 0
    assert len(result.passages) == DEFAULT_TOP_K_RERANK


async def test_no_reranker_keeps_retrieval_order(
    fake_deps: RagDependencies, parameters: WorkflowParameters
) -> None:
    deps = dataclasses.replace(fake_deps, reranker=NoReranker())
    workflow = RagWorkflow(deps, parameters)
    result = await workflow.run(VACATION_QUESTION, mode="retrieve")

    [vector] = await deps.embedder.embed([VACATION_QUESTION])
    expected = await deps.vector_store.query(vector, DEFAULT_TOP_K_RETRIEVE)
    first_five = [p.id for p in expected[:5]]
    assert [p.id for p in result.passages] == first_five
    assert all(p.rerank_score is None for p in result.passages)


async def test_non_default_parameters_size_retrieval_rerank_and_prompt(
    fake_deps: RagDependencies,
) -> None:
    deps = dataclasses.replace(fake_deps, reranker=NoReranker())
    prompts = RagPrompts(system="s", user="ASK {question} USING {context}")
    parameters = workflow_parameters(top_k_retrieve=3, top_k_rerank=2, prompts=prompts)

    result = await RagWorkflow(deps, parameters).run("vacation days")

    assert len(result.passages) == 2
    assert result.answer is not None
    # FakeChatModel echoes the first 200 characters of the user message.
    assert result.answer.startswith("[fake-llm] ASK vacation days USING [1] ")


class RecordingReranker(RerankerPort):
    """Keeps retrieval order and records ``(candidates, top_n)`` per call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        self.calls.append((len(passages), top_n))
        return list(passages[:top_n])

    async def ready(self) -> tuple[bool, str]:
        return True, "recording"


async def test_the_reranker_gets_top_k_retrieve_candidates_and_top_k_rerank(
    fake_deps: RagDependencies, parameters: WorkflowParameters
) -> None:
    reranker = RecordingReranker()
    deps = dataclasses.replace(fake_deps, reranker=reranker)
    sized = dataclasses.replace(parameters, top_k_retrieve=7, top_k_rerank=4)

    await RagWorkflow(deps, sized).run(VACATION_QUESTION, mode="retrieve")

    assert reranker.calls == [(7, 4)]


# --- prompt helpers ----------------------------------------------------------------


# --- retrieval modes -------------------------------------------------------------


class RecordingEmbedder(EmbedderPort):
    """Answers a fixed vector and records every text it was asked to embed."""

    model_name = "recording-embedder"

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    async def ready(self) -> tuple[bool, str]:
        return True, self.model_name


class ScriptedStore(VectorStorePort):
    """Answers one fixed ranking per kind of query and records the calls."""

    def __init__(self, dense: Sequence[str], lexical: Sequence[str]) -> None:
        self.dense, self.lexical = list(dense), list(lexical)
        self.calls: list[tuple[str, object, int]] = []

    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        raise AssertionError("the workflow never writes to the store")

    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]:
        self.calls.append(("dense", list(embedding), top_k))
        return [_passage(pid, pid) for pid in self.dense[:top_k]]

    async def lexical_query(self, text: str, top_k: int) -> list[ScoredPassage]:
        self.calls.append(("lexical", text, top_k))
        return [_passage(pid, pid) for pid in self.lexical[:top_k]]

    async def count(self) -> int:
        return len({*self.dense, *self.lexical})

    async def ready(self) -> tuple[bool, str]:
        return True, "scripted"


def scripted_deps(
    fake_deps: RagDependencies, store: ScriptedStore, embedder: RecordingEmbedder
) -> RagDependencies:
    return dataclasses.replace(
        fake_deps, vector_store=store, embedder=embedder, reranker=NoReranker()
    )


async def test_dense_mode_asks_the_store_by_vector_only(
    fake_deps: RagDependencies,
) -> None:
    store, embedder = ScriptedStore(["a", "b"], ["z"]), RecordingEmbedder()
    workflow = RagWorkflow(
        scripted_deps(fake_deps, store, embedder),
        workflow_parameters(top_k_retrieve=2, top_k_rerank=2),
    )

    result = await workflow.run("vacation days", mode="retrieve")

    assert [p.id for p in result.passages] == ["a", "b"]
    assert store.calls == [("dense", [1.0, 0.0], 2)]


async def test_lexical_mode_skips_the_embed_stage_and_asks_by_words(
    fake_deps: RagDependencies,
) -> None:
    store, embedder = ScriptedStore(["a"], ["z", "y"]), RecordingEmbedder()
    workflow = RagWorkflow(
        scripted_deps(fake_deps, store, embedder),
        workflow_parameters(retrieval_mode="lexical", top_k_retrieve=2, top_k_rerank=2),
    )

    result = await workflow.run("vacation days", mode="retrieve")

    assert [p.id for p in result.passages] == ["z", "y"]
    assert store.calls == [("lexical", "vacation days", 2)]
    assert embedder.texts == []  # no vector is needed, so none is computed
    assert result.timings_ms["embed"] == 0.0
    assert result.timings_ms["retrieve"] > 0.0


async def test_hybrid_mode_asks_both_ways_and_fuses_by_rank(
    fake_deps: RagDependencies,
) -> None:
    store, embedder = ScriptedStore(["a", "b", "c"], ["b", "d"]), RecordingEmbedder()
    workflow = RagWorkflow(
        scripted_deps(fake_deps, store, embedder),
        workflow_parameters(retrieval_mode="hybrid", top_k_retrieve=3, top_k_rerank=3),
    )

    result = await workflow.run("vacation days", mode="retrieve")

    # b is in both rankings; a and d each lead one; c is cut by top_k_retrieve.
    assert [p.id for p in result.passages] == ["b", "a", "d"]
    assert sorted(store.calls, key=lambda call: call[0]) == [
        ("dense", [1.0, 0.0], 3),
        ("lexical", "vacation days", 3),
    ]


async def test_the_query_prefix_goes_in_front_of_the_embedded_question_only(
    fake_deps: RagDependencies,
) -> None:
    store, embedder = ScriptedStore(["a"], ["a"]), RecordingEmbedder()
    workflow = RagWorkflow(
        scripted_deps(fake_deps, store, embedder),
        workflow_parameters(retrieval_mode="hybrid", query_prefix="query: "),
    )

    result = await workflow.run("vacation days", mode="retrieve")

    assert embedder.texts == ["query: vacation days"]
    lexical_calls = [call for call in store.calls if call[0] == "lexical"]
    assert lexical_calls == [("lexical", "vacation days", 20)]  # words, unprefixed
    assert result.question == "vacation days"


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
async def test_every_retrieval_mode_answers_over_the_fake_store(
    fake_deps: RagDependencies, mode: str
) -> None:
    workflow = RagWorkflow(fake_deps, workflow_parameters(retrieval_mode=mode))

    result = await workflow.run("How many vacation days do employees accrue?")

    assert len(result.passages) == 5
    assert "vacation" in result.passages[0].text.lower()
    assert result.answer is not None


def _passage(passage_id: str, text: str) -> ScoredPassage:
    return ScoredPassage(id=passage_id, text=text, retrieval_score=1.0)


def test_rag_prompts_number_the_passages_into_the_user_template() -> None:
    prompts = RagPrompts(system="Be brief.", user="C:\n{context}\nQ: {question}")
    passages = [_passage("a", "Alpha text."), _passage("b", "Beta text.")]

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

    messages = prompts.messages("what is {x}?", [_passage("a", "set {y} to 1")])

    assert messages[1]["content"] == '[1] set {y} to 1|what is {x}?|{"k": 1}'


def test_merge_timings_merges_dicts_right_wins() -> None:
    assert merge_timings({"embed": 1.0}, {"retrieve": 2.0}) == {
        "embed": 1.0,
        "retrieve": 2.0,
    }
    assert merge_timings({"embed": 1.0}, {"embed": 3.0}) == {"embed": 3.0}
    assert merge_timings({}, {}) == {}
