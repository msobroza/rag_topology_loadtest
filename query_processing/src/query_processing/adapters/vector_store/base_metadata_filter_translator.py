"""The port of a filter translator: a metadata filter in a store's own terms.

Each vector store has one translator, which turns our conditions into the
filter its search takes: clauses for Elasticsearch, a ``where`` for Chroma, a
predicate over a passage for the in-memory store. This module imports the
domain only.

Example:
    >>> keep = InMemoryFilterTranslator().translate(
    ...     [MetadataCondition(attribute="topic", comparator="eq", value="pay")])
    >>> keep(passage)
    True
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Generic, TypeVar

from ...domain.retrieval.metadata_filter import MetadataCondition

StoreFilterT = TypeVar("StoreFilterT")


class BaseMetadataFilterTranslator(abc.ABC, Generic[StoreFilterT]):
    """Turns a metadata filter into the filter one store's search takes."""

    @abc.abstractmethod
    def translate(self, conditions: Sequence[MetadataCondition]) -> StoreFilterT:
        """Return the store's filter that keeps what meets every condition.

        Args:
            conditions: The filter's conditions, all to hold; none filters
                nothing.

        Returns:
            The store's own filter.

        Example:
            >>> InMemoryFilterTranslator().translate([])(passage)
            True
        """
