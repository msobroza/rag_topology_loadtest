"""What a load-test user sends: the request pool and the plan of each request."""

from __future__ import annotations

import random

import pytest

from loadtest_settings_helpers import settings_from
from rag_load_test.loadtest.request_plans import (
    RequestPlan,
    RequestPool,
    build_request_pool,
    plan_embeddings_request,
    plan_rag_request,
    plan_rerank_request,
)

# --- what to send ---


POOL = RequestPool(
    questions=("q-one", "q-two", "q-three"),
    rerank_documents=("doc a", "doc b", "doc c", "doc d"),
    top_n=2,
)


def test_build_request_pool_sizes_questions_and_candidates() -> None:
    pool = build_request_pool(question_count=9, candidates=7, top_n=3)

    assert len(pool.questions) == 9
    assert all(pool.questions)
    assert len(pool.rerank_documents) == 7
    assert all(pool.rerank_documents)
    assert pool.top_n == 3


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
def test_plan_rag_request_asks_one_seeded_question(endpoint: str) -> None:
    plan = plan_rag_request(endpoint, POOL, random.Random(7))
    question = random.Random(7).choice(POOL.questions)
    assert plan == RequestPlan(endpoint, {"question": question}, endpoint)


def test_plan_rerank_request_matches_the_workflow_rerank_stage() -> None:
    plan = plan_rerank_request("bge-reranker-base", POOL, random.Random(7))
    assert (plan.path, plan.stats_name) == ("/v3/rerank", "/v3/rerank")
    assert plan.payload == {
        "model": "bge-reranker-base",
        "query": random.Random(7).choice(POOL.questions),
        "documents": ["doc a", "doc b", "doc c", "doc d"],
        "top_n": 2,
    }


def test_plan_embeddings_request_sends_one_question_in_a_list() -> None:
    plan = plan_embeddings_request("bge-small-en-v1.5", POOL, random.Random(7))
    assert (plan.path, plan.stats_name) == ("/v3/embeddings", "/v3/embeddings")
    assert plan.payload == {
        "model": "bge-small-en-v1.5",
        "input": [random.Random(7).choice(POOL.questions)],
    }


def test_default_pool_has_fifty_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = settings_from(monkeypatch, {})
    pool = build_request_pool(settings.question_count, 20, 5)
    assert len(pool.questions) == 50
