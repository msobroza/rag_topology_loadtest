"""The in-memory store's filter translator: a predicate over a record's metadata.

A record that lacks a condition's attribute is dropped, whatever the
comparator, as on a store whose records lack the field (spec 7). A value that
cannot be ordered against the condition's is dropped too.

Example:
    >>> keep = InMemoryFilterTranslator().translate(
    ...     [MetadataCondition(attribute="sequence", comparator="lt", value=3)])
    >>> keep(VectorStoreRecord(text="t", metadata={"id": "p", "sequence": 2}))
    True
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any

from query_processing_core.retrieval.metadata_filter import (
    Comparator,
    MetadataCondition,
)
from query_processing_core.retrieval.vector_store_record import VectorStoreRecord

from ..base_metadata_filter_translator import BaseMetadataFilterTranslator

VectorStoreRecordPredicate = Callable[[VectorStoreRecord], bool]

# Each comparator as a test of (the record's value, the condition's value).
_COMPARE: dict[Comparator, Callable[[Any, Any], bool]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "in": lambda found, values: found in values,
    "nin": lambda found, values: found not in values,
}


class InMemoryFilterTranslator(
    BaseMetadataFilterTranslator[VectorStoreRecordPredicate]
):
    """Turns a metadata filter into a predicate the in-memory store applies."""

    def translate(
        self, conditions: Sequence[MetadataCondition]
    ) -> VectorStoreRecordPredicate:
        """Return a predicate true of a record that meets every condition.

        Args:
            conditions: The filter's conditions; none keeps every record.

        Returns:
            The predicate.

        Example:
            >>> InMemoryFilterTranslator().translate([])(record)
            True
        """
        kept = tuple(conditions)
        return lambda record: all(_meets(record, c) for c in kept)


def _meets(record: VectorStoreRecord, condition: MetadataCondition) -> bool:
    if condition.attribute not in record.metadata:
        return False
    found = record.metadata[condition.attribute]
    try:
        return bool(_COMPARE[condition.comparator](found, condition.value))
    except TypeError:  # a string ordered against a number, say
        return False
