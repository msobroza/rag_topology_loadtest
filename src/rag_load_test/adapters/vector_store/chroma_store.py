"""The Chroma vector store: embedded (a folder) or a Chroma server (a URL).

Open-source Chroma has no ranked text search: it filters documents by
``$contains`` / ``$regex`` only, and its Search API (sparse index) is Chroma Cloud
only; embedded and self-hosted Chroma 1.5.9 refuse it. So this store is a
``BaseVectorStore`` only, and lexical or hybrid retrieval over it is refused at
start-up (``RagSettings.check_retrieval_store``).

chromadb's client is synchronous, so every call runs in a worker thread; the
heavy import lives inside ``open`` so importing this module stays cheap.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

from ...domain.errors import RagDependencyError
from ...domain.passages import MetadataValue, Passage, ScoredPassage
from ...domain.ports import BaseVectorStore

# Cosine distance, so ``1 - distance`` is the cosine similarity ``query`` reports.
_COSINE_SPACE = {"hnsw": {"space": "cosine"}}
_DEFAULT_PORTS = {"http": 80, "https": 443}  # a URL without a port means these


class ChromaVectorStore(BaseVectorStore):
    """BaseVectorStore on a Chroma collection; it ranks no text.

    Any chromadb failure surfaces as ``RagDependencyError("vector_store", ...)``
    with the collection name as its target.

    Args:
        collection: A ``chromadb`` collection.

    Example:
        >>> store = await ChromaVectorStore.open(
        ...     url="", path="chroma", collection="rag_passages")
        >>> hits = await store.query(vector, top_k=20)
    """

    def __init__(self, collection: Any) -> None:
        self._collection = collection
        self._name: str = collection.name

    @classmethod
    async def open(cls, *, url: str, path: str, collection: str) -> ChromaVectorStore:
        """Open the collection, creating it when missing.

        Args:
            url: A Chroma server; "" opens the embedded store at ``path``.
            path: The embedded store's folder.
            collection: The collection the passages live in, created (cosine
                distance) when missing.

        Returns:
            The store on that collection.
        """
        import chromadb

        def connect() -> Any:
            client = _chroma_client(chromadb, url, path)
            return client.get_or_create_collection(
                collection, configuration=_COSINE_SPACE, embedding_function=None
            )

        return cls(await _off_loop(collection, connect))

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
