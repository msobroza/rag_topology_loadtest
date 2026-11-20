"""Elasticsearch's filter translator: a metadata filter as query clauses.

LangChain's store keeps a passage's metadata under ``metadata``, so a condition
on ``source`` filters ``metadata.source``. Dynamic mapping makes a string a
``text`` field with a ``keyword`` sub-field, and only the sub-field matches a
whole value, so a string is matched on ``metadata.<attribute>.keyword``. A
passage that lacks a condition's attribute is dropped, whatever the comparator,
as on the in-memory store.

The clauses are a filter: they score nothing. The vector search gets them as its
``knn`` filter, which the server applies during the search, and the text search
in its ``bool`` query's ``filter``.

Example:
    >>> ElasticsearchFilterTranslator().translate(
    ...     [MetadataCondition(attribute="source", comparator="eq", value="AMF")])
    [{'term': {'metadata.source.keyword': 'AMF'}}]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from query_processing_core.retrieval.metadata_filter import (
    LIST_COMPARATORS,
    MetadataCondition,
)

from ..base_metadata_filter_translator import BaseMetadataFilterTranslator

EsClause = dict[str, Any]

# The comparators an Elasticsearch ``range`` clause names as they are.
_RANGES = frozenset({"gt", "gte", "lt", "lte"})
# The comparators that keep what does not match: a ``must_not``.
_NEGATED = frozenset({"ne", "nin"})


class ElasticsearchFilterTranslator(BaseMetadataFilterTranslator[list[EsClause]]):
    """Turns a metadata filter into the filter clauses of an Elasticsearch query."""

    def translate(self, conditions: Sequence[MetadataCondition]) -> list[EsClause]:
        """Return one filter clause per condition, all to hold.

        Args:
            conditions: The filter's conditions; none gives no clause.

        Returns:
            The clauses, in the conditions' order.

        Example:
            >>> ElasticsearchFilterTranslator().translate(
            ...     [MetadataCondition(attribute="sequence", comparator="gte", value=2)]
            ... )
            [{'range': {'metadata.sequence': {'gte': 2}}}]
        """
        return [_clause(condition) for condition in conditions]


def _clause(condition: MetadataCondition) -> EsClause:
    field, comparator = _field(condition), condition.comparator
    if comparator in _RANGES:
        return {"range": {field: {comparator: condition.value}}}
    term = "terms" if comparator in LIST_COMPARATORS else "term"
    match = {term: {field: condition.value}}
    if comparator not in _NEGATED:
        return match
    # must_not alone keeps a passage that lacks the field; spec 7 and the
    # in-memory store drop it, so the field must exist.
    exists = {"exists": {"field": f"metadata.{condition.attribute}"}}
    return {"bool": {"filter": [exists], "must_not": [match]}}


def _field(condition: MetadataCondition) -> str:
    # A string, or a list of strings, is matched whole on the keyword sub-field.
    value = condition.value
    first = value[0] if isinstance(value, list) and value else value
    suffix = ".keyword" if isinstance(first, str) else ""
    return f"metadata.{condition.attribute}{suffix}"
