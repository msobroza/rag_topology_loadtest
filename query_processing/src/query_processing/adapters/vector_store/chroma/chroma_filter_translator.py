"""Chroma's filter translator: a metadata filter as a ``where``.

Chroma's operators are ours with a ``$`` in front. One condition is a ``where``
of its own; Chroma refuses an ``$and`` of fewer than two, so only several are
joined under it. No condition gives no ``where`` at all, as Chroma refuses an
empty one. Chroma applies the ``where`` before its vector search.

Unlike the in-memory store, Chroma keeps a record that lacks the attribute of
an ``ne`` or ``nin`` condition: it has no operator that asks for a field.

Example:
    >>> ChromaFilterTranslator().translate(
    ...     [MetadataCondition(attribute="active", comparator="eq", value=True)])
    {'active': {'$eq': True}}
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from query_processing_core.retrieval.metadata_filter import MetadataCondition

from ..base_metadata_filter_translator import BaseMetadataFilterTranslator

ChromaWhere = dict[str, Any]


class ChromaFilterTranslator(BaseMetadataFilterTranslator[ChromaWhere | None]):
    """Turns a metadata filter into the ``where`` of a Chroma query."""

    def translate(self, conditions: Sequence[MetadataCondition]) -> ChromaWhere | None:
        """Return the ``where`` that keeps what meets every condition.

        Args:
            conditions: The filter's conditions; none gives ``None``.

        Returns:
            The ``where``, or ``None`` to filter nothing.

        Example:
            >>> ChromaFilterTranslator().translate([])  # None
        """
        wheres = [{c.attribute: {f"${c.comparator}": c.value}} for c in conditions]
        if len(wheres) > 1:
            return {"$and": wheres}
        return wheres[0] if wheres else None
