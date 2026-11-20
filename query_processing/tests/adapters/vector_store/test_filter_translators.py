"""Each store's translator turns a metadata filter into the store's own filter."""

from __future__ import annotations

from typing import Any

import pytest
from query_processing_core.retrieval.metadata_filter import (
    Comparator,
    MetadataCondition,
    MetadataValue,
)
from query_processing_core.retrieval.vector_store_record import VectorStoreRecord

from query_processing.adapters.vector_store.chroma.chroma_filter_translator import (
    ChromaFilterTranslator,
)
from query_processing.adapters.vector_store.elasticsearch import (
    elasticsearch_filter_translator,
)
from query_processing.adapters.vector_store.fake.in_memory_filter_translator import (
    InMemoryFilterTranslator,
)

ElasticsearchFilterTranslator = (
    elasticsearch_filter_translator.ElasticsearchFilterTranslator
)


def cond(
    attribute: str, comparator: Comparator, value: MetadataValue | list[MetadataValue]
) -> MetadataCondition:
    """A condition, written as a test reads it."""
    return MetadataCondition(attribute=attribute, comparator=comparator, value=value)


PASSAGE = VectorStoreRecord(
    text="t", metadata={"id": "p", "source": "AMF", "chunk": 2, "active": True}
)


@pytest.mark.parametrize(
    ("condition", "kept"),
    [
        (cond("source", "eq", "AMF"), True),
        (cond("source", "eq", "GAA"), False),
        (cond("source", "ne", "GAA"), True),
        (cond("source", "ne", "AMF"), False),
        (cond("chunk", "gt", 1), True),
        (cond("chunk", "gt", 2), False),
        (cond("chunk", "gte", 2), True),
        (cond("chunk", "gte", 3), False),
        (cond("chunk", "lt", 3), True),
        (cond("chunk", "lt", 2), False),
        (cond("chunk", "lte", 2), True),
        (cond("chunk", "lte", 1), False),
        (cond("source", "in", ["GAA", "AMF"]), True),
        (cond("source", "in", ["GAA"]), False),
        (cond("source", "nin", ["GAA"]), True),
        (cond("source", "nin", ["AMF"]), False),
        (cond("active", "eq", True), True),
    ],
)
def test_the_in_memory_predicate_keeps_a_passage_that_meets_the_condition(
    condition: MetadataCondition, kept: bool
) -> None:
    assert InMemoryFilterTranslator().translate([condition])(PASSAGE) is kept


@pytest.mark.parametrize(
    "condition",
    [cond("topic", "eq", "x"), cond("topic", "ne", "x"), cond("topic", "nin", ["x"])],
)
def test_the_in_memory_predicate_drops_a_passage_that_lacks_the_field(
    condition: MetadataCondition,
) -> None:
    assert InMemoryFilterTranslator().translate([condition])(PASSAGE) is False


def test_the_in_memory_predicate_drops_a_value_it_cannot_order() -> None:
    assert (
        InMemoryFilterTranslator().translate([cond("source", "gt", 1)])(PASSAGE)
        is False
    )


def test_the_in_memory_predicate_needs_every_condition() -> None:
    translate = InMemoryFilterTranslator().translate

    assert translate([])(PASSAGE) is True
    assert translate([cond("source", "eq", "AMF"), cond("chunk", "lt", 3)])(PASSAGE)
    assert not translate([cond("source", "eq", "AMF"), cond("chunk", "lt", 2)])(PASSAGE)


# --- Elasticsearch ------------------------------------------------------------


@pytest.mark.parametrize(
    ("condition", "clause"),
    [
        (cond("source", "eq", "AMF"), {"term": {"metadata.source.keyword": "AMF"}}),
        (cond("active", "eq", True), {"term": {"metadata.active": True}}),
        (
            cond("source", "ne", "CHAD"),
            {
                "bool": {
                    "filter": [{"exists": {"field": "metadata.source"}}],
                    "must_not": [{"term": {"metadata.source.keyword": "CHAD"}}],
                }
            },
        ),
        (cond("chunk", "gt", 1), {"range": {"metadata.chunk": {"gt": 1}}}),
        (cond("chunk", "gte", 2), {"range": {"metadata.chunk": {"gte": 2}}}),
        (cond("chunk", "lt", 3.5), {"range": {"metadata.chunk": {"lt": 3.5}}}),
        (cond("chunk", "lte", 2), {"range": {"metadata.chunk": {"lte": 2}}}),
        (
            cond("source", "in", ["AMF", "GAA"]),
            {"terms": {"metadata.source.keyword": ["AMF", "GAA"]}},
        ),
        (cond("chunk", "in", [1, 2]), {"terms": {"metadata.chunk": [1, 2]}}),
        (
            cond("source", "nin", ["GAA"]),
            {
                "bool": {
                    "filter": [{"exists": {"field": "metadata.source"}}],
                    "must_not": [{"terms": {"metadata.source.keyword": ["GAA"]}}],
                }
            },
        ),
        (
            cond("on", "nin", [True]),
            {
                "bool": {
                    "filter": [{"exists": {"field": "metadata.on"}}],
                    "must_not": [{"terms": {"metadata.on": [True]}}],
                }
            },
        ),
    ],
)
def test_elasticsearch_turns_each_comparator_into_its_clause(
    condition: MetadataCondition, clause: dict[str, Any]
) -> None:
    assert ElasticsearchFilterTranslator().translate([condition]) == [clause]


def test_elasticsearch_gives_one_clause_per_condition_and_none_for_none() -> None:
    translate = ElasticsearchFilterTranslator().translate

    assert translate([]) == []
    assert translate([cond("source", "in", ["AMF"]), cond("chunk", "gte", 2)]) == [
        {"terms": {"metadata.source.keyword": ["AMF"]}},
        {"range": {"metadata.chunk": {"gte": 2}}},
    ]


# --- Chroma -------------------------------------------------------------------


def test_chroma_gives_no_where_for_no_condition() -> None:
    assert ChromaFilterTranslator().translate([]) is None


@pytest.mark.parametrize(
    ("condition", "where"),
    [
        (cond("active", "eq", True), {"active": {"$eq": True}}),
        (cond("source", "ne", "AMF"), {"source": {"$ne": "AMF"}}),
        (cond("chunk", "gt", 1), {"chunk": {"$gt": 1}}),
        (cond("chunk", "gte", 2), {"chunk": {"$gte": 2}}),
        (cond("chunk", "lt", 3.5), {"chunk": {"$lt": 3.5}}),
        (cond("chunk", "lte", 2), {"chunk": {"$lte": 2}}),
        (cond("source", "in", ["AMF"]), {"source": {"$in": ["AMF"]}}),
        (cond("source", "nin", ["GAA"]), {"source": {"$nin": ["GAA"]}}),
    ],
)
def test_chroma_turns_one_condition_into_its_where(
    condition: MetadataCondition, where: dict[str, Any]
) -> None:
    assert ChromaFilterTranslator().translate([condition]) == where


def test_chroma_joins_several_conditions_under_and() -> None:
    assert ChromaFilterTranslator().translate(
        [cond("source", "in", ["AMF"]), cond("chunk", "lt", 3)]
    ) == {"$and": [{"source": {"$in": ["AMF"]}}, {"chunk": {"$lt": 3}}]}
