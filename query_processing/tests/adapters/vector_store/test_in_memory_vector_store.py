"""The ``fake`` vector store: in memory, with a word ranking for lexical search."""

from __future__ import annotations

import pytest
from query_processing_core.retrieval.metadata_filter import MetadataCondition
from query_processing_core.retrieval.vector_store_record import VectorStoreRecord

from query_processing.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
    distinct_words,
)
from query_processing.domain.components.component_requests import SearchRequest
from query_processing.domain.retrieval.retrieval_mode import RetrievalMode


async def test_in_memory_store_upserts_idempotently_and_queries_in_order() -> None:
    store = InMemoryVectorStore()
    doc = VectorStoreRecord(
        text="one", metadata={"id": "p1", "page": 3, "src": "a.pdf"}
    )
    await store.upsert([doc], [[1.0, 0.0]])
    await store.upsert([doc], [[1.0, 0.0]])  # same id twice -> one row
    await store.upsert(
        [
            VectorStoreRecord(text="two", metadata={"id": "p2"}),
            VectorStoreRecord(text="three", metadata={"id": "p3"}),
        ],
        [[0.0, 1.0], [0.6, 0.8]],
    )

    assert await store.count() == 3
    hits = await store.query([1.0, 0.0], top_k=2)
    assert [(h.id, h.retrieval_score) for h in hits] == [("p1", 1.0), ("p3", 0.6)]
    # Stored whole and read whole: the id, the store's key, is in the metadata.
    assert hits[0].metadata == {"id": "p1", "page": 3, "src": "a.pdf"}


async def test_in_memory_store_ranks_by_direction_not_length() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            VectorStoreRecord(text="a", metadata={"id": "long-off"}),
            VectorStoreRecord(text="b", metadata={"id": "short-on"}),
        ],
        [[10.0, 10.0], [0.3, 0.0]],
    )

    hits = await store.query([2.0, 0.0], top_k=2)

    assert [h.id for h in hits] == ["short-on", "long-off"]
    assert hits[0].retrieval_score == pytest.approx(1.0)
    assert hits[1].retrieval_score == pytest.approx(2**-0.5)


async def test_in_memory_store_finds_passages_by_their_words() -> None:
    store = InMemoryVectorStore()
    passages = [
        VectorStoreRecord(
            text="Employees accrue vacation days monthly", metadata={"id": "vacation"}
        ),
        VectorStoreRecord(
            text="Expense receipts are required", metadata={"id": "expenses"}
        ),
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
    passages = [
        VectorStoreRecord(text=text, metadata={"id": pid})
        for pid, text in texts.items()
    ]
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
    await store.upsert(
        [VectorStoreRecord(text="vacation days", metadata={"id": "p"})], [[1.0, 0.0]]
    )
    await store.upsert(
        [VectorStoreRecord(text="expense receipts", metadata={"id": "p"})], [[1.0, 0.0]]
    )

    assert await store.lexical_query("vacation", top_k=5) == []
    assert [hit.id for hit in await store.lexical_query("expense", top_k=5)] == ["p"]


def test_distinct_words_lower_cases_and_drops_repeats_and_punctuation() -> None:
    assert distinct_words("Vacation, vacation DAYS!") == {"vacation", "days"}
    assert distinct_words("  ...  ") == set()


async def test_in_memory_store_is_ready_once_it_holds_a_passage() -> None:
    store = InMemoryVectorStore()
    assert await store.ready() == (False, "0 passages in memory")

    await store.upsert(
        [VectorStoreRecord(text="one", metadata={"id": "p1"})], [[1.0, 0.0]]
    )
    assert await store.ready() == (True, "1 passages in memory")


async def test_in_memory_store_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="shorter"):  # zip(strict=True)
        await InMemoryVectorStore().upsert(
            [
                VectorStoreRecord(text="a", metadata={"id": "a"}),
                VectorStoreRecord(text="b", metadata={"id": "b"}),
            ],
            [[1.0]],
        )


TOPICS = {"pay-1": "pay", "leave-1": "leave", "pay-2": "pay", "untagged": None}


async def store_with_topics() -> InMemoryVectorStore:
    """A store of four passages on vacation pay, three of them with a topic."""
    store = InMemoryVectorStore()
    passages = [
        VectorStoreRecord(
            text="vacation pay",
            metadata={"id": pid, **({"topic": topic} if topic else {})},
        )
        for pid, topic in TOPICS.items()
    ]
    await store.upsert(passages, [[1.0, 0.0]] * len(passages))
    return store


@pytest.mark.parametrize("retrieval_mode", ["dense", "lexical", "hybrid"])
async def test_a_search_returns_only_the_passages_that_meet_every_condition(
    retrieval_mode: RetrievalMode,
) -> None:
    store = await store_with_topics()

    answer = await store.ainvoke(
        SearchRequest(
            embedding=[1.0, 0.0],
            text="vacation",
            top_k=5,
            retrieval_mode=retrieval_mode,
            rrf_k=60,
            metadata_filter=[
                MetadataCondition(attribute="topic", comparator="eq", value="pay")
            ],
        )
    )

    assert sorted(p.id for p in answer.passages) == ["pay-1", "pay-2"]


async def test_the_filter_applies_before_top_k() -> None:
    store = await store_with_topics()
    leave = [MetadataCondition(attribute="topic", comparator="eq", value="leave")]

    dense = await store.query([1.0, 0.0], top_k=1, metadata_filter=leave)
    lexical = await store.lexical_query("vacation", top_k=1, metadata_filter=leave)

    assert [p.id for p in dense] == [p.id for p in lexical] == ["leave-1"]


async def test_a_search_without_conditions_filters_nothing() -> None:
    store = await store_with_topics()

    answer = await store.ainvoke(
        SearchRequest(embedding=[1.0, 0.0], top_k=5, retrieval_mode="dense")
    )

    assert len(answer.passages) == len(TOPICS)


def cond(attribute: str, comparator: str, value: object) -> MetadataCondition:
    """A condition, built as a client or a settings file sends it."""
    return MetadataCondition.model_validate(
        {"attribute": attribute, "comparator": comparator, "value": value}
    )


SOURCES = {"amf-on": ("AMF", True), "amf-off": ("AMF", False), "gaa-on": ("GAA", True)}


async def store_of_sources() -> InMemoryVectorStore:
    """A store of three passages, by source and by whether they are active."""
    store = InMemoryVectorStore()
    passages = [
        VectorStoreRecord(
            text="vacation", metadata={"id": pid, "source": src, "active": on}
        )
        for pid, (src, on) in SOURCES.items()
    ]
    await store.upsert(passages, [[1.0, 0.0]] * len(passages))
    return store


@pytest.mark.parametrize("retrieval_mode", ["dense", "lexical", "hybrid"])
async def test_the_prefilter_and_the_requests_conditions_must_all_hold(
    retrieval_mode: RetrievalMode,
) -> None:
    store = await store_of_sources()
    store.prefilter = (cond("active", "eq", True),)

    answer = await store.ainvoke(
        SearchRequest(
            embedding=[1.0, 0.0],
            text="vacation",
            metadata_filter=[cond("source", "in", ["AMF"])],
            top_k=5,
            retrieval_mode=retrieval_mode,
            rrf_k=60,
        )
    )

    assert [p.id for p in answer.passages] == ["amf-on"]


async def test_the_prefilter_applies_to_a_search_without_conditions() -> None:
    store = await store_of_sources()
    store.prefilter = (cond("active", "eq", True),)

    answer = await store.ainvoke(
        SearchRequest(embedding=[1.0, 0.0], top_k=5, retrieval_mode="dense")
    )

    assert sorted(p.id for p in answer.passages) == ["amf-on", "gaa-on"]


def test_a_store_states_no_prefilter_by_default() -> None:
    assert tuple(InMemoryVectorStore().prefilter) == ()
