"""The Elasticsearch vector store (LangChain's ``AsyncElasticsearchStore``).

The heavy import lives inside ``open`` so importing this module stays cheap. The
index name comes from the caller (``RAG_ES_INDEX`` via ``RagPorts``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..domain.errors import RagDependencyError
from ..domain.passages import Passage, ScoredPassage
from ..domain.ports import VectorStorePort


class ElasticsearchVectorStore(VectorStorePort):
    """VectorStorePort on LangChain's ``AsyncElasticsearchStore``.

    Vectors come from our own EmbedderPort, so the store is driven through
    ``aadd_embeddings`` and ``asimilarity_search_by_vector_with_relevance_scores``
    and never needs a LangChain ``Embeddings`` object. Any Elasticsearch failure
    surfaces as ``RagDependencyError("vector_store", ...)``.

    Example::

        store = ElasticsearchVectorStore.open("http://localhost:9200", index="rag")
        hits = await store.query(vector, top_k=20)
    """

    def __init__(self, store: Any, index: str) -> None:
        self._store, self.index = store, index

    @classmethod
    def open(
        cls, url: str, *, index: str, api_key: str = ""
    ) -> ElasticsearchVectorStore:
        from langchain_elasticsearch import AsyncElasticsearchStore

        store = AsyncElasticsearchStore(index, es_url=url, es_api_key=api_key or None)
        return cls(store, index)

    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if not passages:
            return
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

    async def count(self) -> int:
        client = self._store.client
        if not await self._call(client.indices.exists, index=self.index):
            return 0
        return int((await self._call(client.count, index=self.index))["count"])

    async def ready(self) -> tuple[bool, str]:
        count = await self.count()
        return count > 0, f"{count} passages in {self.index}"

    async def aclose(self) -> None:
        """Close the Elasticsearch connections held by the underlying store."""
        await self._store.aclose()

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
