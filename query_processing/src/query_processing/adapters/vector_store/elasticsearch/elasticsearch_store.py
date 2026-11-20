"""The Elasticsearch vector store (LangChain's ``AsyncElasticsearchStore``).

The heavy import lives inside ``open`` so importing this module stays cheap. The
index name comes from the caller (its ``elasticsearch`` block, via the composition).

This adapter creates a missing index itself, just before the first write, from
the mapping LangChain's own retrieval strategy produces (its
``es_mappings_settings``) sized from the first vector. LangChain cannot: it is
given no embedding and no ``num_dimensions``, so its own creation raises
"retrieval strategy requires embeddings". An existing index is left alone.

The text analyzer (the block's ``text_analyzer``) is set when the index is made and
an index keeps it. Any analyzer but ``standard`` is mapped on the text field, so
indexing and the BM25 ``match`` of ``lexical_query`` both use it. With
``standard`` (or "") the text field is left unmapped: Elasticsearch's dynamic
mapping then gives it exactly what it got before this adapter created indices.

Its API key comes from a key source, when its block names one: a call that
Elasticsearch refuses for its key (its 401) opens a new client with the key
the source then gives and is made once more, a single further attempt
(``elasticsearch_client_key.py``). The replaced client is closed with the
store, not at once, so a call still in flight on it gets its answer. Every
call goes through ``_call``, so each one, a write and the readiness check
included, does so.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from functools import partial
from typing import Any

from ....domain.errors import RagDependencyError, describe_error
from ....domain.retrieval.metadata_filter import MetadataCondition
from ....domain.retrieval.passages import Passage, ScoredPassage
from ...key_source.base_key_source import BaseKeySource
from ..base_vector_store import BaseLexicalSearch, BaseVectorStore
from .elasticsearch_client_key import ElasticsearchClientKey, StoreConnector
from .elasticsearch_filter_translator import ElasticsearchFilterTranslator

# One call of LangChain's store, given the store in use when it is made.
StoreRequest = Callable[[Any], Awaitable[Any]]


class ElasticsearchVectorStore(BaseVectorStore, BaseLexicalSearch):
    """Both store ports on LangChain's ``AsyncElasticsearchStore``.

    Elasticsearch ranks text itself (BM25), so this one store is the workflow's
    ``BaseVectorStore`` and its ``BaseLexicalSearch``.

    Vectors come from our own BaseEmbedder, so the store is driven through
    ``aadd_embeddings`` and ``asimilarity_search_by_vector_with_relevance_scores``
    and never needs a LangChain ``Embeddings`` object. Any Elasticsearch failure
    surfaces as ``RagDependencyError("vector_store", ...)``.

    Args:
        store: An ``AsyncElasticsearchStore`` already bound to ``index``.
        index: The Elasticsearch index the passages live in.
        text_analyzer: The analyzer of the passages' text; "" or ``standard``
            leaves Elasticsearch's default.
        index_strategy: The retrieval strategy ``store`` was built with; a new
            index gets its mapping.
        key: Builds ``store`` again with its key source's next key when
            Elasticsearch refuses the one it holds; None: a refusal fails at once.

    Example:
        >>> store = await ElasticsearchVectorStore.open(
        ...     "http://localhost:9200", index="rag", key_source=None)
        >>> hits = await store.query(vector, top_k=20)
    """

    def __init__(
        self,
        store: Any,
        index: str,
        text_analyzer: str = "",
        *,
        index_strategy: Any,
        key: ElasticsearchClientKey | None = None,
    ) -> None:
        self._store, self.index = store, index
        self._text_analyzer = "" if text_analyzer == "standard" else text_analyzer
        self._index_strategy = index_strategy
        self._index_ensured = False
        self._filter_translator = ElasticsearchFilterTranslator()
        self._key = key
        # One new client for calls refused together, not one each.
        self._reconnecting = asyncio.Lock()
        # Clients a refused key replaced, closed with the store: a call still in
        # flight on one gets its answer.
        self._replaced: list[Any] = []

    @classmethod
    async def open(
        cls,
        url: str,
        *,
        index: str,
        key_source: BaseKeySource | None = None,
        text_analyzer: str = "",
    ) -> ElasticsearchVectorStore:
        """Build the store on a new ``AsyncElasticsearchStore``, importing it here.

        Args:
            url: The Elasticsearch URL; ``user:password@`` in it for basic
                authentication.
            index: The index the passages live in (``elasticsearch.index``).
            key_source: Where its Elasticsearch API key comes from; None
                connects without one.
            text_analyzer: The analyzer of the passages' text; "" or ``standard``
                leaves Elasticsearch's default.

        Returns:
            The vector store on that index.

        Raises:
            RagDependencyError: If the key source could not get its key.

        Example:
            >>> store = await ElasticsearchVectorStore.open(url, index="rag")
        """
        from elasticsearch.helpers.vectorstore import (
            AsyncDenseVectorStrategy,
            DistanceMetric,
        )

        # What LangChain's default ApproxRetrievalStrategy() converts to, made
        # explicit so the index this adapter creates gets the same mapping.
        strategy = AsyncDenseVectorStrategy(
            distance=DistanceMetric.COSINE, hybrid=False, rrf=False
        )
        connect = partial(_langchain_store, url, index, strategy)
        return await cls.open_with(
            connect,
            index,
            text_analyzer,
            index_strategy=strategy,
            key_source=key_source,
        )

    @classmethod
    async def open_with(
        cls,
        connect: StoreConnector,
        index: str,
        text_analyzer: str = "",
        *,
        index_strategy: Any,
        key_source: BaseKeySource | None,
    ) -> ElasticsearchVectorStore:
        """Build the store on what ``connect`` builds from the key source's key.

        ``open`` gives LangChain's own; a test gives one that builds its fake.

        Args:
            connect: Builds LangChain's store from a key, or with none.
            index: The index the passages live in.
            text_analyzer: The analyzer of the passages' text.
            index_strategy: The retrieval strategy ``connect``'s store is built
                with; a new index gets its mapping.
            key_source: Where its API key comes from; None sends none.

        Returns:
            The vector store, opened again with a new key when one is refused.

        Example:
            >>> store = await ElasticsearchVectorStore.open_with(
            ...     connect, "rag", index_strategy=strategy, key_source=source)
        """
        key = ElasticsearchClientKey(connect, key_source)
        return cls(
            await key.connected(),
            index,
            text_analyzer,
            index_strategy=index_strategy,
            key=key,
        )

    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Store passages with their vectors, keyed by passage id.

        Args:
            passages: The passages to store; none is a no-op.
            embeddings: One vector per passage, in the same order.
        """
        if not passages:
            return
        await self._ensure_index(len(embeddings[0]))
        pairs = [(p.text, list(e)) for p, e in zip(passages, embeddings, strict=True)]
        metadatas, ids = [dict(p.metadata) for p in passages], [p.id for p in passages]
        await self._call(
            lambda store: store.aadd_embeddings(pairs, metadatas=metadatas, ids=ids)
        )

    async def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> list[ScoredPassage]:
        """Find the passages nearest to a vector among those the filter keeps.

        Args:
            embedding: The query vector.
            top_k: How many passages to return at most.
            metadata_filter: Conditions a passage must all meet, sent as the
                ``knn`` clause's filter, which the server applies during the
                search; none filters nothing.

        Returns:
            The nearest passages with Elasticsearch's relevance score as
            ``retrieval_score``.
        """
        clauses = self._filter_translator.translate(metadata_filter)
        hits = await self._call(
            lambda store: store.asimilarity_search_by_vector_with_relevance_scores(
                list(embedding), k=top_k, filter=clauses, doc_builder=_document_with_id
            )
        )
        return [
            ScoredPassage(
                id=doc.id or "",
                text=doc.page_content,
                metadata=doc.metadata,
                retrieval_score=float(score),
            )
            for doc, score in hits
        ]

    async def lexical_query(
        self,
        text: str,
        top_k: int,
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> list[ScoredPassage]:
        """Find the passages whose words best match a text (Elasticsearch BM25).

        Args:
            text: The text searched for.
            top_k: How many passages to return at most.
            metadata_filter: Conditions a passage must all meet, sent in the
                ``filter`` of a ``bool`` query, where they score nothing; none
                filters nothing.

        Returns:
            The best-matching passages, best first, with Elasticsearch's BM25
            ``_score`` as ``retrieval_score``.
        """
        query = self._text_query(text, metadata_filter)
        found = await self._call(
            lambda store: store.client.search(index=self.index, query=query, size=top_k)
        )
        return [
            ScoredPassage(
                id=hit["_id"],
                text=hit["_source"].get("text", ""),
                metadata=hit["_source"].get("metadata", {}),
                retrieval_score=float(hit["_score"]),
            )
            for hit in found["hits"]["hits"]
        ]

    def _text_query(
        self, text: str, metadata_filter: Sequence[MetadataCondition]
    ) -> dict[str, Any]:
        # LangChain's store keeps the passage text in the ``text`` field.
        match = {"match": {"text": text}}
        clauses = self._filter_translator.translate(metadata_filter)
        return {"bool": {"must": match, "filter": clauses}} if clauses else match

    async def count(self) -> int:
        """Count the passages in the index.

        Returns:
            The number of documents in the index; 0 when it does not exist yet.
        """
        if not await self._index_exists():
            return 0
        counted = await self._call(lambda store: store.client.count(index=self.index))
        return int(counted["count"])

    async def ready(self) -> tuple[bool, str]:
        """Report whether the index holds any passage.

        Returns:
            Whether the index is non-empty, and how many passages it holds; when
            its text analyzer differs from the configured one, also that.
        """
        count = await self.count()
        status = f"{count} passages in {self.index}"
        if count and self._text_analyzer:
            status += await self._analyzer_mismatch()
        return count > 0, status

    async def aclose(self) -> None:
        """Close the client in use and every client a refused key replaced."""
        for client in [*self._replaced, self._store]:
            await client.aclose()
        self._replaced.clear()

    async def _ensure_index(self, dimensions: int) -> None:
        # Create a missing index before LangChain's first write would try to and
        # fail; once per store, as an index is never dropped.
        if self._index_ensured:
            return
        if not await self._index_exists():
            body = await self._index_body(dimensions)
            await self._call(
                lambda store: store.client.indices.create(index=self.index, **body)
            )
        self._index_ensured = True

    async def _index_exists(self) -> bool:
        exists = await self._call(
            lambda store: store.client.indices.exists(index=self.index)
        )
        return bool(exists)

    async def _index_body(self, dimensions: int) -> dict[str, Any]:
        # LangChain's path (_create_index_if_not_exists in elasticsearch-py's
        # vectorstore helper) with the dimensions it lacks, plus any analyzer.
        text_field = self._store.query_field
        fields = {
            "text_field": text_field,
            "vector_field": self._store.vector_query_field,
        }
        mappings, settings = self._index_strategy.es_mappings_settings(
            **fields, num_dimensions=dimensions
        )
        if self._text_analyzer:  # standard: left to dynamic mapping, as before
            mappings["properties"][text_field] = {
                "type": "text",
                "analyzer": self._text_analyzer,
            }
        strategy_setup = self._index_strategy.before_index_creation
        await self._call(lambda store: strategy_setup(client=store.client, **fields))
        return {"mappings": mappings, "settings": settings}

    async def _analyzer_mismatch(self) -> str:
        # "" when the index's text field has the configured analyzer, else a note.
        found = await self._call(
            lambda store: store.client.indices.get_mapping(index=self.index)
        )
        text_field = self._store.query_field
        analyzers = {
            found[name]["mappings"]
            .get("properties", {})
            .get(text_field, {})
            .get("analyzer", "standard")
            for name in found
        }
        if analyzers <= {self._text_analyzer}:
            return ""
        actual = ", ".join(repr(a) for a in sorted(analyzers))
        return (
            f"; its text analyzer is {actual}, not the configured "
            f"{self._text_analyzer!r}: an index keeps the analyzer it was created with"
        )

    async def _call(self, request: StoreRequest) -> Any:
        # Every call that reaches Elasticsearch: a key refused (its 401) opens a
        # new client with the source's next key, for one further attempt.
        store = self._store
        try:
            return await request(store)
        except Exception as exc:
            if self._key is None or not self._key.rotates_on(exc):
                raise self._failure(exc) from exc
        await self._reconnect(refused_on=store)
        try:
            return await request(self._store)
        except Exception as exc:
            raise self._failure(exc) from exc

    async def _reconnect(self, *, refused_on: Any) -> None:
        # A client another call replaced already is not replaced again.
        async with self._reconnecting:
            if self._store is not refused_on or self._key is None:
                return
            self._store = await self._key.reconnected()
            self._replaced.append(refused_on)

    def _failure(self, exc: Exception) -> RagDependencyError:
        detail = self._key.described(exc) if self._key else describe_error(exc)
        return RagDependencyError("vector_store", detail, target=self.index)


def _langchain_store(url: str, index: str, strategy: Any, key: str | None) -> Any:
    # LangChain's store, its client sending ``key`` as an Elasticsearch API key.
    from langchain_elasticsearch import AsyncElasticsearchStore

    return AsyncElasticsearchStore(index, es_url=url, es_api_key=key, strategy=strategy)


def _document_with_id(hit: dict[str, Any]) -> Any:
    # The default builder drops Elasticsearch's _id, which is our passage id.
    from langchain_core.documents import Document

    source = hit["_source"]
    return Document(
        id=hit["_id"],
        page_content=source.get("text", ""),
        metadata=source.get("metadata", {}),
    )
