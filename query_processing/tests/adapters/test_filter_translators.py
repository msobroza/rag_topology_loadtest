"""Each store's translator turns a metadata filter into the store's own filter."""

from __future__ import annotations

import pytest

from query_processing.adapters.vector_store.fake.in_memory_filter_translator import (
    InMemoryFilterTranslator,
)
from query_processing.domain.metadata_filter import Comparator, MetadataCondition
from query_processing.domain.passages import MetadataValue, Passage


def cond(
    attribute: str, comparator: Comparator, value: MetadataValue | list[MetadataValue]
) -> MetadataCondition:
    """A condition, written as a test reads it."""
    return MetadataCondition(attribute=attribute, comparator=comparator, value=value)


PASSAGE = Passage(
    id="p", text="t", metadata={"source": "AMF", "chunk": 2, "active": True}
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
