"""The ingest use case: embed passages in batches and upsert them."""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.passages import Passage
from ..domain.ports import EmbedderPort, VectorStorePort


async def ingest(
    passages: Sequence[Passage],
    embedder: EmbedderPort,
    store: VectorStorePort,
    *,
    batch_size: int = 64,
) -> int:
    """Embed passages in batches and upsert them; returns the number upserted."""
    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        await store.upsert(batch, await embedder.embed([p.text for p in batch]))
    return len(passages)
