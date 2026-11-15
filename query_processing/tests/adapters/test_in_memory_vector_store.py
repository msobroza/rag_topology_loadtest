"""The ``fake`` vector store: in memory, with a word ranking for lexical search."""

from __future__ import annotations

import pytest

from rag_load_test.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
    distinct_words,
)
from rag_load_test.domain.passages import Passage


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


async def test_in_memory_store_finds_passages_by_their_words() -> None:
    store = InMemoryVectorStore()
    passages = [
        Passage(id="vacation", text="Employees accrue vacation days monthly"),
        Passage(id="expenses", text="Expense receipts are required"),
    ]
    await store.upsert(passages, [[1.0, 0.0], [0.0, 1.0]])

    hits = await store.lexical_query("vacation days", top_k=5)

    assert [hit.id for hit in hits] == ["vacation"]  # "expenses" shares no word
    assert hits[0].text == "Employees accrue vacation days monthly"
    assert hits[0].retrieval_score == 2.0  # two distinct query words found


async def test_in_memory_store_ranks_by_shared_words_ties_in_insertion_order() -> None:
    store = InMemoryVectorStore()
    texts = {
        "one": "Vacation policy",
        "none": "expense receipts",
        "two-first": "vacation days, vacation days",
        "three": "VACATION days policy",
        "two-second": "days of vacation",
    }
    passages = [Passage(id=pid, text=text) for pid, text in texts.items()]
    await store.upsert(passages, [[1.0, 0.0]] * len(passages))

    hits = await store.lexical_query("vacation vacation days policy", top_k=10)

    assert [(h.id, h.retrieval_score) for h in hits] == [
        ("three", 3.0),
        ("one", 2.0),  # the three ties keep the order they were stored in
        ("two-first", 2.0),  # repeated words count once
        ("two-second", 2.0),
    ]
    assert [h.id for h in await store.lexical_query("days policy", top_k=2)] == [
        "three",
        "one",
    ]


async def test_in_memory_store_reindexes_the_words_of_an_overwritten_passage() -> None:
    store = InMemoryVectorStore()
    await store.upsert([Passage(id="p", text="vacation days")], [[1.0, 0.0]])
    await store.upsert([Passage(id="p", text="expense receipts")], [[1.0, 0.0]])

    assert await store.lexical_query("vacation", top_k=5) == []
    assert [hit.id for hit in await store.lexical_query("expense", top_k=5)] == ["p"]


def test_distinct_words_lower_cases_and_drops_repeats_and_punctuation() -> None:
    assert distinct_words("Vacation, vacation DAYS!") == {"vacation", "days"}
    assert distinct_words("  ...  ") == set()


async def test_in_memory_store_is_ready_once_it_holds_a_passage() -> None:
    store = InMemoryVectorStore()
    assert await store.ready() == (False, "0 passages in memory")

    await store.upsert([Passage(id="p1", text="one")], [[1.0, 0.0]])
    assert await store.ready() == (True, "1 passages in memory")


async def test_in_memory_store_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="shorter"):  # zip(strict=True)
        await InMemoryVectorStore().upsert(
            [Passage(id="a", text="a"), Passage(id="b", text="b")], [[1.0]]
        )
