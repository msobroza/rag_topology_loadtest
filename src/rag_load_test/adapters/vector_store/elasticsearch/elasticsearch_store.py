"""The Elasticsearch vector store (LangChain's ``AsyncElasticsearchStore``).

The heavy import lives inside ``open`` so importing this module stays cheap. The
index name comes from the caller (``RAG_ES_INDEX`` via ``RagPorts``).

This adapter creates a missing index itself, just before the first write, from
the mapping LangChain's own retrieval strategy produces (its
``es_mappings_settings``) sized from the first vector. LangChain cannot: it is
given no embedding and no ``num_dimensions``, so its own creation raises
"retrieval strategy requires embeddings". An existing index is left alone.

The text analyzer (``RAG_ES_TEXT_ANALYZER``) is set when the index is created and
an index keeps it. Any analyzer but ``standard`` is mapped on the text field, so
indexing and the BM25 ``match`` of ``lexical_query`` both use it. With
``standard`` (or "") the text field is left unmapped: Elasticsearch's dynamic
mapping then gives it exactly what it got before this adapter created indices.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ....domain.errors import RagDependencyError
from ....domain.passages import Passage, ScoredPassage
from ..base_vector_store import BaseLexicalSearch, BaseVectorStore


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

    Example:
        >>> store = ElasticsearchVectorStore.open("http://localhost:9200", index="rag")
        >>> hits = await store.query(vector, top_k=20)
    """

    def __init__(
        self,
        store: Any,
        index: str,
        text_analyzer: str = "",
        *,
        index_strategy: Any,
    ) -> None:
        self._store, self.index = store, index
        self._text_analyzer = "" if text_analyzer == "standard" else text_analyzer
        self._index_strategy = index_strategy
        self._index_ensured = False

    @classmethod
    def open(
        cls, url: str, *, index: str, api_key: str = "", text_analyzer: str = ""
    ) -> ElasticsearchVectorStore:
        """Build the store on a new ``AsyncElasticsearchStore``, importing it here.

        Args:
            url: The Elasticsearch URL.
            index: The index the passages live in (``RAG_ES_INDEX``).
            api_key: An Elasticsearch API key; "" connects without one.
            text_analyzer: The analyzer of the passages' text; "" or ``standard``
                leaves Elasticsearch's default.

        Returns:
            The vector store on that index.
        """
        from elasticsearch.helpers.vectorstore import (
            AsyncDenseVectorStrategy,
            DistanceMetric,
        )
        from langchain_elasticsearch import AsyncElasticsearchStore

        # What LangChain's default ApproxRetrievalStrategy() converts to, made
        # explicit so the index this adapter creates gets the same mapping.
        strategy = AsyncDenseVectorStrategy(
            distance=DistanceMetric.COSINE, hybrid=False, rrf=False
        )
        store = AsyncElasticsearchStore(
            index, es_url=url, es_api_key=api_key or None, strategy=strategy
        )
        return cls(store, index, text_analyzer, index_strategy=strategy)

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
        await self._call(
            self._store.aadd_embeddings,
            pairs,
            metadatas=[dict(p.metadata) for p in passages],
            ids=[p.id for p in passages],
        )

    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]:
        """Find the passages nearest to a vector.

        Args:
            embedding: The query vector.
            top_k: How many passages to return at most.

        Returns:
            The nearest passages with Elasticsearch's relevance score as
            ``retrieval_score``.
        """
        hits = await self._call(
            self._store.asimilarity_search_by_vector_with_relevance_scores,
            list(embedding),
            k=top_k,
            doc_builder=_document_with_id,
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

    async def lexical_query(self, text: str, top_k: int) -> list[ScoredPassage]:
        """Find the passages whose words best match a text (Elasticsearch BM25).

        Args:
            text: The text searched for.
            top_k: How many passages to return at most.

        Returns:
            The best-matching passages, best first, with Elasticsearch's BM25
            ``_score`` as ``retrieval_score``.
        """
        # LangChain's store keeps the passage text in the ``text`` field.
        found = await self._call(
            self._store.client.search,
            index=self.index,
            query={"match": {"text": text}},
            size=top_k,
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

    async def count(self) -> int:
        """Count the passages in the index.

        Returns:
            The number of documents in the index; 0 when it does not exist yet.
        """
        client = self._store.client
        if not await self._call(client.indices.exists, index=self.index):
            return 0
        return int((await self._call(client.count, index=self.index))["count"])

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
        """Close the Elasticsearch connections held by the underlying store."""
        await self._store.aclose()

    async def _ensure_index(self, dimensions: int) -> None:
        # Create a missing index before LangChain's first write would try to and
        # fail; once per store, as an index is never dropped.
        if self._index_ensured:
            return
        client = self._store.client
        if not await self._call(client.indices.exists, index=self.index):
            body = await self._index_body(dimensions)
            await self._call(client.indices.create, index=self.index, **body)
        self._index_ensured = True

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
        await self._call(strategy_setup, client=self._store.client, **fields)
        return {"mappings": mappings, "settings": settings}

    async def _analyzer_mismatch(self) -> str:
        # "" when the index's text field has the configured analyzer, else a note.
        found = await self._call(
            self._store.client.indices.get_mapping, index=self.index
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

    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            raise RagDependencyError("vector_store", detail, target=self.index) from exc


def _document_with_id(hit: dict[str, Any]) -> Any:
    # The default builder drops Elasticsearch's _id, which is our passage id.
    from langchain_core.documents import Document

    source = hit["_source"]
    return Document(
        id=hit["_id"],
        page_content=source.get("text", ""),
        metadata=source.get("metadata", {}),
    )
