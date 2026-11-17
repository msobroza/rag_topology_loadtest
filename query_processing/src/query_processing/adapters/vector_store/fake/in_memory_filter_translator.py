"""The in-memory store's filter translator: a predicate over a passage's metadata.

A passage that lacks a condition's attribute is dropped, whatever the
comparator, as on a store whose passages lack the field (spec 7). A value that
cannot be ordered against the condition's is dropped too.

Example:
    >>> keep = InMemoryFilterTranslator().translate(
    ...     [MetadataCondition(attribute="chunk", comparator="lt", value=3)])
    >>> keep(Passage(id="p", text="t", metadata={"chunk": 2}))
    True
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import Any

from ....domain.metadata_filter import Comparator, MetadataCondition
from ....domain.passages import Passage
from ..base_metadata_filter_translator import BaseMetadataFilterTranslator

PassagePredicate = Callable[[Passage], bool]

# Each comparator as a test of (the passage's value, the condition's value).
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


class InMemoryFilterTranslator(BaseMetadataFilterTranslator[PassagePredicate]):
    """Turns a metadata filter into a predicate the in-memory store applies."""

    def translate(self, conditions: Sequence[MetadataCondition]) -> PassagePredicate:
        """Return a predicate true of a passage that meets every condition.

        Args:
            conditions: The filter's conditions; none keeps every passage.

        Returns:
            The predicate.

        Example:
            >>> InMemoryFilterTranslator().translate([])(passage)
            True
        """
        kept = tuple(conditions)
        return lambda passage: all(_meets(passage, c) for c in kept)


def _meets(passage: Passage, condition: MetadataCondition) -> bool:
    if condition.attribute not in passage.metadata:
        return False
    found = passage.metadata[condition.attribute]
    try:
        return bool(_COMPARE[condition.comparator](found, condition.value))
    except TypeError:  # a string ordered against a number, say
        return False
