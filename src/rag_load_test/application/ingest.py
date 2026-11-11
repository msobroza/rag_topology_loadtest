"""The ingest use case: embed passages in batches and upsert them."""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.passages import Passage
from ..domain.ports import BaseEmbedder, BaseVectorStore


async def ingest(
    passages: Sequence[Passage],
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    *,
    batch_size: int = 64,
) -> int:
    """Embed passages in batches and upsert them.

    Args:
        passages: The passages to store.
        embedder: Embeds each batch.
        store: Receives each batch with its vectors.
        batch_size: Passages per embed and upsert call.

    Returns:
        The number of passages upserted.
    """
    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        await store.upsert(batch, await embedder.embed([p.text for p in batch]))
    return len(passages)
