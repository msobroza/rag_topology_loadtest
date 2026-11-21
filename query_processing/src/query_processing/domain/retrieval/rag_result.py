"""The outcome of a RAG run, the keeping of the best-scored records, and their ids.

The records themselves are the core's vector-store records, and the mode and
the name of the total timing are the core's response body's (ADR 0014). A
record keeps its id in its metadata, which may hold none: every place that
needs the id asks ``required_record_id``, which refuses such a record in words.

Nothing heavy is imported here, so the OVMS-only and fake topologies never pay
for torch or elasticsearch.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field
from query_processing_core.query_response import RagMode
from query_processing_core.retrieval.metadata_filter import MetadataCondition
from query_processing_core.retrieval.vector_store_record import (
    ScoredVectorStoreRecord,
    VectorStoreRecord,
)


class RagResult(BaseModel):
    """Outcome of one workflow run.

    ``timings_ms`` has one key per stage of the graph, in the order of its
    description, then ``TOTAL_TIMING_KEY``. ``tokens`` has one key per stage
    whose LLM reported both counts in this run, in the same order, each
    ``{"prompt": n, "completion": m}``; empty when none did, so a reader never
    mistakes a missing count for 0.

    Example:
        >>> RagResult(question="q", mode="retrieve", answer=None, records=[],
        ...           timings_ms={}).tokens
        {}
    """

    question: str
    mode: RagMode
    answer: str | None
    records: list[ScoredVectorStoreRecord]
    timings_ms: dict[str, float]
    tokens: dict[str, dict[str, int]] = Field(default_factory=dict)
    # The conditions applied: the client's, then those the stages added.
    metadata_filter: list[MetadataCondition] = Field(default_factory=list)


def top_by_score(
    records: Sequence[ScoredVectorStoreRecord], scores: Sequence[float], top_n: int
) -> list[ScoredVectorStoreRecord]:
    """Keep the ``top_n`` vector-store records with the highest score.

    Args:
        records: The candidates to rank.
        scores: One score per record, in the same order.
        top_n: How many records to keep.

    Returns:
        Copies of the best records, highest score first, each with its score
        stamped as ``rerank_score``.

    Example:
        >>> top_by_score([a, b, c], [0.1, 0.9, 0.5], 2)
        [b(0.9), c(0.5)]
    """
    ranked = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)
    return [
        records[i].model_copy(update={"rerank_score": float(scores[i])})
        for i in ranked[:top_n]
    ]


def required_record_id(record: VectorStoreRecord) -> str:
    """Return a record's id, ``metadata["id"]``, or refuse a record that holds none.

    doclang's record accepts a metadata with no id and reads it only when asked,
    so a store's write, rank fusion and the rerank route ask here first: a bare
    ``KeyError: 'id'`` names neither the record nor what it lacks.

    Args:
        record: A vector-store record, scored or not.

    Returns:
        Its id, as text.

    Raises:
        ValueError: Its metadata holds no ``id``; the message quotes the start
            of its text and the keys its metadata holds.

    Example:
        >>> required_record_id(VectorStoreRecord(text="t", metadata={"id": "p1"}))
        'p1'
    """
    if "id" in record.metadata:
        return record.id
    raise ValueError(
        f"the vector-store record {record.text[:40]!r} holds no id; expected a "
        'text id under metadata["id"], but its metadata holds only '
        f"{sorted(record.metadata)}"
    )
