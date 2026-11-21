"""The ingest use case: embed vector-store records in batches and upsert them."""

from __future__ import annotations

from collections.abc import Sequence

from query_processing_core.retrieval.vector_store_record import VectorStoreRecord

from ..adapters.embedder.base_embedder import BaseEmbedder
from ..adapters.vector_store.base_vector_store import BaseVectorStore


async def ingest(
    records: Sequence[VectorStoreRecord],
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    *,
    batch_size: int = 64,
) -> int:
    """Embed records in batches, as documents with no query prefix, and upsert them.

    Args:
        records: The vector-store records to store.
        embedder: Embeds each batch.
        store: Receives each batch with its vectors.
        batch_size: Records per embed and upsert call.

    Returns:
        The number of records upserted.
    """
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        vectors = await embedder.embed_documents([p.text for p in batch])
        await store.upsert(batch, vectors)
    return len(records)
