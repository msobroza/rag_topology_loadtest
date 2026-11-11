"""The Chroma vector store: embedded (a folder) or a Chroma server (a URL).

Open-source Chroma has no ranked text search: it filters documents by
``$contains`` / ``$regex`` only, and its Search API is Chroma Cloud only. So
``lexical_query`` ranks with a ``Bm25Index`` kept in this process, filled from the
collection by ``open`` and by every ``upsert`` made through this store. Passages
another process ingests after ``open`` are not seen by ``lexical_query`` until
this process restarts (``query`` sees them at once).

chromadb's client is synchronous, so every call runs in a worker thread; the
heavy import lives inside ``open`` so importing this module stays cheap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

from ...domain.bm25 import Bm25Index
from ...domain.errors import RagDependencyError
from ...domain.passages import MetadataValue, Passage, ScoredPassage
from ...domain.ports import VectorStorePort

# Cosine distance, so ``1 - distance`` is the cosine similarity ``query`` reports.
_COSINE_SPACE = {"hnsw": {"space": "cosine"}}
_DEFAULT_PORTS = {"http": 80, "https": 443}  # a URL without a port means these


class ChromaVectorStore(VectorStorePort):
    """VectorStorePort on a Chroma collection, with BM25 kept in this process.

    Any chromadb failure surfaces as ``RagDependencyError("vector_store", ...)``
    with the collection name as its target.

    Args:
        collection: A ``chromadb`` collection.
        lexical_index: The BM25 index that answers ``lexical_query``.

    Example:
        >>> store = await ChromaVectorStore.open(
        ...     url="", path="chroma", collection="rag_passages",
        ...     lexical_index=Bm25Index(k1=1.2, b=0.75))
        >>> hits = await store.lexical_query("vacation days", top_k=20)
    """

    def __init__(self, collection: Any, lexical_index: Bm25Index) -> None:
        self._collection, self._lexical_index = collection, lexical_index
        self._name: str = collection.name

    @classmethod
    async def open(
        cls, *, url: str, path: str, collection: str, lexical_index: Bm25Index
    ) -> ChromaVectorStore:
        """Open the collection and index its passages for lexical search.

        Args:
            url: A Chroma server; "" opens the embedded store at ``path``.
            path: The embedded store's folder.
            collection: The collection the passages live in, created (cosine
                distance) when missing.
            lexical_index: An empty BM25 index, filled from the collection.

        Returns:
            The store on that collection, its BM25 index holding every passage
            already in it.
        """
        import chromadb

        def connect() -> tuple[Any, int]:
            client = _chroma_client(chromadb, url, path)
            found = client.get_or_create_collection(
                collection, configuration=_COSINE_SPACE, embedding_function=None
            )
            return found, client.get_max_batch_size()

        found, page_size = await _off_loop(collection, connect)
        store = cls(found, lexical_index)
        await store._index_existing(page_size)
        return store

    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Store passages with their vectors; a known id is replaced whole.

        Args:
            passages: The passages to store; none is a no-op.
            embeddings: One vector per passage, in the same order.
        """
        if not passages:
            return
        ids = [p.id for p in passages]
        vectors = [list(e) for _, e in zip(passages, embeddings, strict=True)]

        def replace() -> None:
            # Chroma's own upsert merges new metadata into the old, so a key a
            # re-ingested passage dropped would survive: delete, then add.
            self._collection.delete(ids=ids)
            self._collection.add(
                ids=ids,
                embeddings=vectors,
                documents=[p.text for p in passages],
                metadatas=[_chroma_metadata(p.metadata) for p in passages],
            )

        await self._call(replace)
        for passage in passages:
            self._lexical_index.add(passage.id, passage.text)

    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]:
        """Find the passages nearest to a vector.

        Args:
            embedding: The query vector.
            top_k: How many passages to return at most.

        Returns:
            The nearest passages, nearest first, with the cosine similarity
            (``1 - cosine distance``, higher is nearer) as ``retrieval_score``.
        """
        if top_k <= 0:
            return []
        found = await self._call(
            self._collection.query,
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        rows = zip(
            found["ids"][0],
            found["documents"][0],
            found["metadatas"][0],
            found["distances"][0],
            strict=True,
        )
        return [_scored(pid, text, meta, 1.0 - dist) for pid, text, meta, dist in rows]

    async def lexical_query(self, text: str, top_k: int) -> list[ScoredPassage]:
        """Find the passages whose words best match a text, by BM25.

        Args:
            text: The text searched for.
            top_k: How many passages to return at most.

        Returns:
            The best-matching passages, best first, with the BM25 score as
            ``retrieval_score``; a passage sharing no word with ``text`` is left
            out.
        """
        ranked = self._lexical_index.search(text, top_k)
        if not ranked:
            return []  # Chroma rejects get(ids=[])
        found = await self._call(
            self._collection.get,
            ids=[pid for pid, _ in ranked],
            include=["documents", "metadatas"],
        )
        # get() returns the ids in storage order, not in the order asked for.
        columns = zip(found["ids"], found["documents"], found["metadatas"], strict=True)
        rows = {pid: (text, meta) for pid, text, meta in columns}
        return [_scored(pid, *rows[pid], score) for pid, score in ranked if pid in rows]

    async def count(self) -> int:
        """Count the stored passages.

        Returns:
            The number of passages in the collection.
        """
        return int(await self._call(self._collection.count))

    async def ready(self) -> tuple[bool, str]:
        """Report whether the collection holds any passage.

        Returns:
            Whether the collection is non-empty, and how many passages it holds.
        """
        count = await self.count()
        return count > 0, f"{count} passages in {self._name}"

    async def _index_existing(self, page_size: int) -> None:
        # Pages of the server's own max batch size: the largest it will send.
        offset = 0
        while True:
            page = await self._call(
                self._collection.get,
                include=["documents"],
                limit=page_size,
                offset=offset,
            )
            for pid, text in zip(page["ids"], page["documents"] or [], strict=True):
                self._lexical_index.add(pid, text or "")
            if len(page["ids"]) < page_size:
                return
            offset += page_size

    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return await _off_loop(self._name, fn, *args, **kwargs)


async def _off_loop(
    target: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    # chromadb blocks (disk or HTTP), so it never runs on the event loop.
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        raise RagDependencyError("vector_store", detail, target=target) from exc


def _chroma_client(chromadb: Any, url: str, path: str) -> Any:
    # Telemetry off: a load test must not phone home from every worker.
    settings = chromadb.Settings(anonymized_telemetry=False)
    if not url:
        return chromadb.PersistentClient(path=path, settings=settings)
    parts = urlsplit(url)
    return chromadb.HttpClient(
        host=parts.hostname or "",
        port=parts.port or _DEFAULT_PORTS.get(parts.scheme, _DEFAULT_PORTS["http"]),
        ssl=parts.scheme == "https",
        settings=settings,
    )


def _chroma_metadata(
    metadata: dict[str, MetadataValue],
) -> dict[str, MetadataValue] | None:
    # Chroma rejects an empty metadata dict but accepts None.
    return dict(metadata) or None


def _scored(
    passage_id: str, text: str | None, metadata: Any, score: float
) -> ScoredPassage:
    return ScoredPassage(
        id=passage_id,
        text=text or "",
        metadata=dict(metadata or {}),
        retrieval_score=float(score),
    )
