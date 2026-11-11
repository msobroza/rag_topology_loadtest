"""The ``fake`` backends: the embedder, the reranker, the in-memory store and the chat model."""

from __future__ import annotations

import math
import time

import pytest

from rag_load_test.adapters.fakes import fake_models
from rag_load_test.adapters.fakes.fake_models import (
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
)
from rag_load_test.domain.passages import Passage, ScoredPassage
from store_fakes import in_memory_store


def _scored(pid: str, text: str = "", retrieval_score: float = 0.5) -> ScoredPassage:
    return ScoredPassage(id=pid, text=text or pid, retrieval_score=retrieval_score)


# --- fakes -------------------------------------------------------------------


async def test_fake_embedder_is_deterministic_unit_norm_and_topical() -> None:
    embedder = FakeEmbedder()
    pie, pie_again, tart, physics = await embedder.embed(
        ["red apple pie", "red apple pie", "red apple tart", "quantum gravity lecture"]
    )

    assert pie == pie_again
    assert math.isclose(math.sqrt(sum(x * x for x in pie)), 1.0, rel_tol=1e-9)
    assert fake_models.cosine(pie, tart) > fake_models.cosine(pie, physics)


async def test_fake_reranker_scores_by_overlap_and_truncates() -> None:
    passages = [
        _scored("off", "gardening tips for spring"),
        _scored("hit", "the elasticsearch cluster shards"),
        _scored("near", "cluster of stars"),
    ]
    kept = await FakeReranker().rerank("elasticsearch cluster", passages, top_n=2)

    assert [p.id for p in kept] == ["hit", "near"]
    assert kept[0].rerank_score is not None
    assert kept[0].rerank_score > 0
    assert kept[0].rerank_score > (kept[1].rerank_score or 0)


async def test_in_memory_store_upserts_idempotently_and_queries_in_order() -> None:
    store = in_memory_store()
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


async def test_in_memory_store_finds_passages_by_their_words() -> None:
    store = in_memory_store()
    passages = [
        Passage(id="vacation", text="Employees accrue vacation days monthly"),
        Passage(id="expenses", text="Expense receipts are required"),
    ]
    await store.upsert(passages, [[1.0, 0.0], [0.0, 1.0]])

    hits = await store.lexical_query("vacation days", top_k=5)

    assert [hit.id for hit in hits] == ["vacation"]  # "expenses" shares no word
    assert hits[0].text == "Employees accrue vacation days monthly"
    assert hits[0].retrieval_score > 0


async def test_in_memory_store_reindexes_the_words_of_an_overwritten_passage() -> None:
    store = in_memory_store()
    await store.upsert([Passage(id="p", text="vacation days")], [[1.0, 0.0]])
    await store.upsert([Passage(id="p", text="expense receipts")], [[1.0, 0.0]])

    assert await store.lexical_query("vacation", top_k=5) == []
    assert [hit.id for hit in await store.lexical_query("expense", top_k=5)] == ["p"]


async def test_in_memory_store_is_ready_once_it_holds_a_passage() -> None:
    store = in_memory_store()
    assert await store.ready() == (False, "0 passages in memory")

    await store.upsert([Passage(id="p1", text="one")], [[1.0, 0.0]])
    assert await store.ready() == (True, "1 passages in memory")


async def test_in_memory_store_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="shorter"):  # zip(strict=True)
        await in_memory_store().upsert(
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
