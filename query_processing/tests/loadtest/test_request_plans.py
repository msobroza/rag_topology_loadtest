"""What a load-test user sends: the request pool and the plan of each request."""

from __future__ import annotations

import random

import pytest

from loadtest_settings_helpers import settings_from
from query_processing.loadtest.request_plans import (
    RequestPlan,
    RequestPool,
    build_request_pool,
    plan_embeddings_request,
    plan_rag_request,
    plan_rerank_request,
)

# --- what to send ---


FOLLOW_UP = {
    "question": "q-two",
    "conversational_history": [{"role": "user", "content": "q-one"}],
    "metadata_filter": [{"attribute": "topic", "comparator": "eq", "value": "x"}],
}
POOL = RequestPool(
    questions=({"question": "q-one"}, FOLLOW_UP, {"question": "q-three"}),
    rerank_documents=("doc a", "doc b", "doc c", "doc d"),
    top_n=2,
)


def test_build_request_pool_sizes_questions_and_candidates() -> None:
    pool = build_request_pool(question_count=9, candidates=7, top_n=3)

    assert len(pool.questions) == 9
    assert all(
        set(body) == {"question"} and body["question"] for body in pool.questions
    )
    assert len(pool.rerank_documents) == 7
    assert all(pool.rerank_documents)
    assert pool.top_n == 3


@pytest.mark.parametrize("endpoint", ["/query", "/retrieve"])
def test_plan_rag_request_asks_one_seeded_question(endpoint: str) -> None:
    plan = plan_rag_request(endpoint, POOL, random.Random(7))
    body = random.Random(7).choice(POOL.questions)
    assert plan == RequestPlan(endpoint, body, endpoint)


def test_plan_rag_request_sends_the_history_and_filter_of_the_body_it_draws() -> None:
    pool = RequestPool(questions=(FOLLOW_UP,), rerank_documents=("d",), top_n=1)
    assert plan_rag_request("/query", pool, random.Random(7)).payload == FOLLOW_UP


def test_a_model_server_request_asks_the_question_of_a_follow_up() -> None:
    pool = RequestPool(questions=(FOLLOW_UP,), rerank_documents=("d",), top_n=1)
    rng = random.Random(7)
    assert plan_rerank_request("r", pool, rng).payload["query"] == "q-two"
    assert plan_embeddings_request("e", pool, rng).payload["input"] == ["q-two"]


def test_plan_rerank_request_matches_the_workflow_rerank_stage() -> None:
    plan = plan_rerank_request("bge-reranker-base", POOL, random.Random(7))
    assert (plan.path, plan.stats_name) == ("/v3/rerank", "/v3/rerank")
    assert plan.payload == {
        "model": "bge-reranker-base",
        "query": random.Random(7).choice(POOL.questions)["question"],
        "documents": ["doc a", "doc b", "doc c", "doc d"],
        "top_n": 2,
    }


def test_plan_embeddings_request_sends_one_question_in_a_list() -> None:
    plan = plan_embeddings_request("bge-small-en-v1.5", POOL, random.Random(7))
    assert (plan.path, plan.stats_name) == ("/v3/embeddings", "/v3/embeddings")
    assert plan.payload == {
        "model": "bge-small-en-v1.5",
        "input": [random.Random(7).choice(POOL.questions)["question"]],
    }


def test_default_pool_has_fifty_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = settings_from(monkeypatch, {})
    pool = build_request_pool(settings.question_count, 20, 5)
    assert len(pool.questions) == 50


# --- a model server answering elsewhere: the path moves, the row's name does not ---
# qp-compare lines rows up by name, so a run on vLLM (/v1/...) lines up with the
# same run on OVMS (/v3/...).


def test_a_rerank_target_posts_to_the_path_the_settings_name() -> None:
    plan = plan_rerank_request("r", POOL, random.Random(7), path="/v1/rerank")
    assert (plan.path, plan.stats_name) == ("/v1/rerank", "/v3/rerank")
    assert plan.payload == plan_rerank_request("r", POOL, random.Random(7)).payload


def test_without_it_a_rerank_target_posts_to_v3() -> None:
    plan = plan_rerank_request("r", POOL, random.Random(7), path="")
    assert (plan.path, plan.stats_name) == ("/v3/rerank", "/v3/rerank")


@pytest.mark.parametrize(
    ("path", "posted"), [("", "/v3/embeddings"), ("/v1/embeddings", "/v1/embeddings")]
)
def test_an_embeddings_target_posts_to_the_path_named_else_v3(
    path: str, posted: str
) -> None:
    plan = plan_embeddings_request("e", POOL, random.Random(7), path=path)
    assert (plan.path, plan.stats_name) == (posted, "/v3/embeddings")
