"""In-memory vector store: the ``fake`` backend.

Vectors are searched by brute-force cosine. Words are ranked by a stand-in, not
BM25: a record scores the number of distinct query words its text contains.
That is enough to exercise the lexical and hybrid paths without Elasticsearch.
A metadata filter is a predicate over each record (``InMemoryFilterTranslator``),
applied before either ranking.

Example:
    >>> store = InMemoryVectorStore()
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from query_processing_core.retrieval.metadata_filter import MetadataCondition
from query_processing_core.retrieval.vector_store_record import (
    ScoredVectorStoreRecord,
    VectorStoreRecord,
)

from ....domain.retrieval.rag_result import required_record_id
from ....infrastructure.fake_text import cosine
from ..base_vector_store import BaseLexicalSearch, BaseVectorStore, record_as_read
from .in_memory_filter_translator import InMemoryFilterTranslator

_WORD = re.compile(r"\w+")


def distinct_words(text: str) -> set[str]:
    """Split a text into its distinct lower-cased word tokens.

    Args:
        text: Any text.

    Returns:
        The set of its word tokens (runs of letters, digits or _),
        lower-cased.

    Example:
        >>> distinct_words("Vacation, vacation days")
        {'vacation', 'days'}
    """
    return set(_WORD.findall(text.lower()))


class InMemoryVectorStore(BaseVectorStore, BaseLexicalSearch):
    """Records and vectors in a dict, searched by cosine or by shared words.

    The lexical ranking is a stand-in for a real text ranking such as BM25: the
    score is how many distinct query words the record's text contains.

    Args:
        rows: The rows the store holds from the start, each record and its
            vector under the id it is held under, as an index ingested before
            holds them; none by default.
    """

    def __init__(
        self,
        rows: Mapping[str, tuple[VectorStoreRecord, Sequence[float]]] | None = None,
    ) -> None:
        self._rows: dict[str, tuple[VectorStoreRecord, list[float]]] = {
            held_under: (record, list(vector))
            for held_under, (record, vector) in (rows or {}).items()
        }
        self._filter_translator = InMemoryFilterTranslator()

    async def upsert(
        self,
        records: Sequence[VectorStoreRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store records whole with their vectors; a known id is overwritten.

        Args:
            records: The records to store.
            embeddings: One vector per record, in the same order.

        Raises:
            ValueError: A record holds no id; nothing is stored.
        """
        ids = [required_record_id(record) for record in records]
        rows = zip(ids, records, embeddings, strict=True)
        for held_under, record, embedding in rows:
            self._rows[held_under] = (record, list(embedding))

    async def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> list[ScoredVectorStoreRecord]:
        """Find the records nearest to a vector by cosine, among those kept.

        Args:
            embedding: The query vector.
            top_k: How many records to return at most.
            metadata_filter: Conditions a record must all meet.

        Returns:
            The nearest records, nearest first, with the cosine as
            ``retrieval_score``.
        """
        scored = sorted(
            (
                (cosine(embedding, vec), held_under, p)
                for held_under, (p, vec) in self._kept(metadata_filter)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            record_as_read(held_under, p.text, p.metadata, score)
            for score, held_under, p in scored[:top_k]
        ]

    async def lexical_query(
        self,
        text: str,
        top_k: int,
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> list[ScoredVectorStoreRecord]:
        """Find the records sharing the most distinct words with a text, of those kept.

        Args:
            text: The text searched for.
            top_k: How many records to return at most.
            metadata_filter: Conditions a record must all meet.

        Returns:
            The records, most shared words first (ties in insertion order),
            with that count as ``retrieval_score``; a record sharing no word
            is left out.
        """
        wanted = distinct_words(text)
        scored = [
            (len(wanted & distinct_words(p.text)), held_under, p)
            for held_under, (p, _) in self._kept(metadata_filter)
        ]
        # sorted() is stable, so equal scores keep the dict's insertion order.
        ranked = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])
        return [
            record_as_read(held_under, p.text, p.metadata, score)
            for score, held_under, p in ranked[:top_k]
        ]

    def _kept(
        self, metadata_filter: Sequence[MetadataCondition]
    ) -> list[tuple[str, tuple[VectorStoreRecord, list[float]]]]:
        # Each row with the key it is held under: a record read has that id,
        # whatever its stored metadata holds (the port's contract).
        keep = self._filter_translator.translate(metadata_filter)
        return [(key, row) for key, row in self._rows.items() if keep(row[0])]

    async def count(self) -> int:
        """Count the stored records.

        Returns:
            The number of records in memory.
        """
        return len(self._rows)

    async def ready(self) -> tuple[bool, str]:
        """Report whether any record is stored.

        Returns:
            Whether the store is non-empty, and how many records it holds.
        """
        return bool(self._rows), f"{len(self._rows)} records in memory"
