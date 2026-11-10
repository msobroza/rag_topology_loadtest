"""Workflow tests: the graph over fake ports, the prompt helpers, and the wiring."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from rag_load_test.adapters import ElasticsearchVectorStore, NoReranker
from rag_load_test.fakes import FakeChatModel, FakeEmbedder, FakeReranker
from rag_load_test.loadtest import QUESTIONS
from rag_load_test.models import STAGES, ScoredPassage
from rag_load_test.ovms import OvmsEmbedder, OvmsReranker
from rag_load_test.settings import RagSettings
from rag_load_test.workflow import (
    SYSTEM_PROMPT,
    TOP_K_RERANK,
    TOP_K_RETRIEVE,
    RagDependencies,
    RagWorkflow,
    build_chat_model,
    build_dependencies,
    build_embedder,
    build_reranker,
    build_vector_store,
    merge_timings,
    rag_messages,
)

VACATION_QUESTION = QUESTIONS[0]
TIMING_KEYS = {*STAGES, "total"}


def _refuse(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP request: {request.method} {request.url}")


@pytest.fixture
async def offline_client() -> AsyncIterator[httpx.AsyncClient]:
    """An httpx client that fails on any request: builders must not call out."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(_refuse)) as client:
        yield client


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(1, thread_name_prefix="rag-test")
    yield pool
    pool.shutdown(wait=False)


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
    # The 30-document corpus holds three vacation documents; all rank first.
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


# --- wiring from settings --------------------------------------------------------


def test_build_embedder_fake(
    fake_settings: RagSettings,
    offline_client: httpx.AsyncClient,
    executor: ThreadPoolExecutor,
) -> None:
    embedder = build_embedder(fake_settings, client=offline_client, executor=executor)
    assert isinstance(embedder, FakeEmbedder)


def test_build_embedder_ovms(
    settings_factory: Callable[..., RagSettings],
    offline_client: httpx.AsyncClient,
    executor: ThreadPoolExecutor,
) -> None:
    settings = settings_factory(embedder_backend="ovms")
    embedder = build_embedder(settings, client=offline_client, executor=executor)
    assert isinstance(embedder, OvmsEmbedder)
    assert embedder.model_name == "ovms:bge-small-en-v1.5"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("fake", FakeReranker), ("none", NoReranker), ("ovms", OvmsReranker)],
)
def test_build_reranker_picks_backend(
    settings_factory: Callable[..., RagSettings],
    offline_client: httpx.AsyncClient,
    executor: ThreadPoolExecutor,
    backend: str,
    expected: type,
) -> None:
    settings = settings_factory(reranker_backend=backend)
    reranker = build_reranker(settings, client=offline_client, executor=executor)
    assert isinstance(reranker, expected)


def test_build_reranker_ovms_model_name(
    settings_factory: Callable[..., RagSettings],
    offline_client: httpx.AsyncClient,
    executor: ThreadPoolExecutor,
) -> None:
    settings = settings_factory(reranker_backend="ovms")
    reranker = build_reranker(settings, client=offline_client, executor=executor)
    assert isinstance(reranker, OvmsReranker)
    assert reranker.model_name == "ovms:bge-reranker-base"


def test_build_chat_model_fake(fake_settings: RagSettings) -> None:
    assert isinstance(build_chat_model(fake_settings), FakeChatModel)


def test_build_vector_store_constructs_without_network(
    settings_factory: Callable[..., RagSettings],
) -> None:
    # Port 9 (discard) would refuse a connection; construction must never try one.
    settings = settings_factory(es_url="http://127.0.0.1:9")
    assert isinstance(build_vector_store(settings), ElasticsearchVectorStore)


def test_build_dependencies_wires_every_port(
    fake_settings: RagSettings,
    offline_client: httpx.AsyncClient,
    executor: ThreadPoolExecutor,
) -> None:
    deps = build_dependencies(fake_settings, client=offline_client, executor=executor)

    assert isinstance(deps.embedder, FakeEmbedder)
    assert isinstance(deps.vector_store, ElasticsearchVectorStore)
    assert isinstance(deps.reranker, FakeReranker)
    assert isinstance(deps.chat_model, FakeChatModel)
