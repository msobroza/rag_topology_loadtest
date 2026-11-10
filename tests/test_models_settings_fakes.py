"""Behavioural tests for the dependency-free core: models, settings and fakes."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from rag_load_test import fakes
from rag_load_test.fakes import (
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
    InMemoryVectorStore,
)
from rag_load_test.models import (
    ChatModelPort,
    EmbedderPort,
    Passage,
    RagDependencyError,
    RerankerPort,
    ScoredPassage,
    VectorStorePort,
    top_by_score,
)
from rag_load_test.settings import RagSettings


def _scored(pid: str, text: str = "", retrieval_score: float = 0.5) -> ScoredPassage:
    return ScoredPassage(id=pid, text=text or pid, retrieval_score=retrieval_score)


# --- models ------------------------------------------------------------------


def test_top_by_score_orders_desc_truncates_and_stamps_float_rerank_score() -> None:
    passages = [_scored("a", retrieval_score=0.1), _scored("b"), _scored("c")]
    kept = top_by_score(passages, scores=[1, 3, 2], top_n=2)  # ints on purpose

    assert [p.id for p in kept] == ["b", "c"]
    assert [p.rerank_score for p in kept] == [3.0, 2.0]
    assert all(type(p.rerank_score) is float for p in kept)
    assert [p.retrieval_score for p in kept] == [0.5, 0.5]
    assert passages[0].rerank_score is None  # inputs are copied, not mutated


def test_rag_dependency_error_message_carries_component_kind_and_target() -> None:
    err = RagDependencyError("reranker", "HTTP 503", target="http://ovms:8001")

    assert err.kind == "unavailable"
    assert str(err) == "reranker unavailable: HTTP 503 (target=http://ovms:8001)"
    assert "timeout" in str(RagDependencyError("llm", "slow", kind="timeout"))


def test_ports_are_runtime_checkable_and_fakes_satisfy_them() -> None:
    assert isinstance(FakeEmbedder(), EmbedderPort)
    assert isinstance(FakeReranker(), RerankerPort)
    assert isinstance(InMemoryVectorStore(), VectorStorePort)
    assert isinstance(FakeChatModel(), ChatModelPort)
    assert not isinstance(FakeChatModel(), VectorStorePort)


# --- settings ----------------------------------------------------------------


def test_settings_defaults(settings_factory: Callable[..., RagSettings]) -> None:
    s = settings_factory()

    assert s.topology == "monolith"
    assert (s.embedder_backend, s.reranker_backend) == ("local", "local")
    assert s.ovms_rerank_url == "http://localhost:8001"
    assert s.ovms_embeddings_url == "http://localhost:8002"
    assert s.es_url == "http://localhost:9200"
    assert s.ovms_auth == ""


def test_settings_read_rag_prefixed_env(
    settings_factory: Callable[..., RagSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_RERANKER_BACKEND", "ovms")
    monkeypatch.setenv("RAG_DOMINO_TRACING", "true")

    s = settings_factory()
    assert s.reranker_backend == "ovms"
    assert s.domino_tracing is True


def test_settings_reject_unknown_backend(
    settings_factory: Callable[..., RagSettings],
) -> None:
    with pytest.raises(ValidationError, match="tpu"):
        settings_factory(embedder_backend="tpu")


def test_settings_field_set_is_exactly_fifteen() -> None:
    assert sorted(RagSettings.model_fields) == [
        "domino_tracing",
        "embedder_backend",
        "embedder_model",
        "es_api_key",
        "es_url",
        "llm_backend",
        "openai_model",
        "ovms_auth",
        "ovms_embeddings_model",
        "ovms_embeddings_url",
        "ovms_rerank_model",
        "ovms_rerank_url",
        "reranker_backend",
        "reranker_model",
        "topology",
    ]


# --- fakes -------------------------------------------------------------------


async def test_fake_embedder_is_deterministic_unit_norm_and_topical() -> None:
    embedder = FakeEmbedder()
    pie, pie_again, tart, physics = await embedder.embed(
        ["red apple pie", "red apple pie", "red apple tart", "quantum gravity lecture"]
    )

    assert pie == pie_again
    assert math.isclose(math.sqrt(sum(x * x for x in pie)), 1.0, rel_tol=1e-9)
    assert fakes.cosine(pie, tart) > fakes.cosine(pie, physics)


async def test_fake_reranker_scores_by_overlap_and_truncates() -> None:
    passages = [
        _scored("off", "gardening tips for spring"),
        _scored("hit", "the elasticsearch cluster shards"),
        _scored("near", "cluster of stars"),
    ]
    kept = await FakeReranker().rerank("elasticsearch cluster", passages, top_n=2)

    assert [p.id for p in kept] == ["hit", "near"]
    assert kept[0].rerank_score is not None and kept[0].rerank_score > 0
    assert kept[0].rerank_score > (kept[1].rerank_score or 0)


async def test_in_memory_store_upserts_idempotently_and_queries_in_order() -> None:
    store = InMemoryVectorStore()
    doc = Passage(id="p1", text="one", metadata={"page": 3, "src": "a.pdf"})
    await store.upsert([doc], [[1.0, 0.0]])
    await store.upsert([doc], [[1.0, 0.0]])  # same id twice -> one row
    await store.upsert(
        [Passage(id="p2", text="two"), Passage(id="p3", text="three")],
        [[0.0, 1.0], [0.6, 0.8]],
    )

    assert await store.count() == 3
    hits = await store.query([1.0, 0.0], top_k=2)
    assert [(h.id, h.retrieval_score) for h in hits] == [("p1", 1.0), ("p3", 0.6)]
    assert hits[0].metadata == {"page": 3, "src": "a.pdf"}


async def test_in_memory_store_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        await InMemoryVectorStore().upsert(
            [Passage(id="a", text="a"), Passage(id="b", text="b")], [[1.0]]
        )


async def test_fake_chat_model_echoes_last_user_message_and_counts_calls() -> None:
    llm = FakeChatModel()
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]

    assert await llm.generate(messages) == "[fake-llm] second"
    assert await llm.generate([{"role": "system", "content": "x"}]) == "[fake-llm] "
    assert llm.calls == 2


async def test_fake_chat_model_honours_latency() -> None:
    llm = FakeChatModel(latency_ms=30)
    start = time.perf_counter()
    await llm.generate([{"role": "user", "content": "hi"}])

    assert time.perf_counter() - start >= 0.025
