"""Workflow tests: the graph over fake ports and the prompt helpers."""

from __future__ import annotations

import dataclasses

from rag_load_test.adapters import NoReranker
from rag_load_test.fakes import FakeChatModel
from rag_load_test.loadtest import QUESTIONS
from rag_load_test.models import (
    STAGES,
    TOP_K_RERANK,
    TOP_K_RETRIEVE,
    RagDependencies,
    ScoredPassage,
)
from rag_load_test.workflow import (
    SYSTEM_PROMPT,
    RagWorkflow,
    merge_timings,
    rag_messages,
)

VACATION_QUESTION = QUESTIONS[0]
TIMING_KEYS = {*STAGES, "total"}


# --- the graph ----------------------------------------------------------------


async def test_query_mode_answers_from_reranked_passages(
    fake_deps: RagDependencies,
) -> None:
    result = await RagWorkflow(fake_deps).run(VACATION_QUESTION)

    assert result.question == VACATION_QUESTION
    assert result.mode == "query"
    assert result.answer is not None and result.answer.startswith("[fake-llm]")
    assert 0 < len(result.passages) <= TOP_K_RERANK
    scores = [p.rerank_score for p in result.passages]
    assert all(score is not None for score in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


async def test_query_mode_times_every_stage(fake_deps: RagDependencies) -> None:
    result = await RagWorkflow(fake_deps).run(VACATION_QUESTION)

    assert set(result.timings_ms) == TIMING_KEYS
    assert result.timings_ms["total"] > 0
    assert all(ms >= 0 for ms in result.timings_ms.values())


async def test_vacation_question_retrieves_vacation_policy(
    fake_deps: RagDependencies,
) -> None:
    result = await RagWorkflow(fake_deps).run(VACATION_QUESTION)

    topics = [p.metadata["topic"] for p in result.passages]
    # One seeded document in ten is a vacation policy; they outrank every other topic.
    assert topics[:3] == ["vacation-policy"] * 3


async def test_retrieve_mode_skips_generation(fake_deps: RagDependencies) -> None:
    result = await RagWorkflow(fake_deps).run(VACATION_QUESTION, mode="retrieve")

    assert isinstance(fake_deps.chat_model, FakeChatModel)
    assert fake_deps.chat_model.calls == 0
    assert result.mode == "retrieve"
    assert result.answer is None
    assert result.timings_ms["generate"] == 0.0
    assert result.timings_ms["total"] > 0
    assert 0 < len(result.passages) <= TOP_K_RERANK


async def test_no_reranker_keeps_retrieval_order(fake_deps: RagDependencies) -> None:
    deps = dataclasses.replace(fake_deps, reranker=NoReranker())
    result = await RagWorkflow(deps).run(VACATION_QUESTION, mode="retrieve")

    [vector] = await deps.embedder.embed([VACATION_QUESTION])
    expected = await deps.vector_store.query(vector, TOP_K_RETRIEVE)
    assert [p.id for p in result.passages] == [p.id for p in expected[:TOP_K_RERANK]]
    assert all(p.rerank_score is None for p in result.passages)


# --- prompt helpers ----------------------------------------------------------------


def _passage(passage_id: str, text: str) -> ScoredPassage:
    return ScoredPassage(id=passage_id, text=text, retrieval_score=1.0)


def test_rag_messages_numbers_passages_and_ends_with_question() -> None:
    passages = [_passage("a", "Alpha text."), _passage("b", "Beta text.")]

    messages = rag_messages("What is alpha?", passages)

    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "[1] Alpha text.\n[2] Beta text." in user
    assert user.endswith("Question: What is alpha?")


def test_merge_timings_merges_dicts_right_wins() -> None:
    assert merge_timings({"embed": 1.0}, {"retrieve": 2.0}) == {
        "embed": 1.0,
        "retrieve": 2.0,
    }
    assert merge_timings({"embed": 1.0}, {"embed": 3.0}) == {"embed": 3.0}
    assert merge_timings({}, {}) == {}
