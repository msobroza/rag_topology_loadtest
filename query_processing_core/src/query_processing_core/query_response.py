"""The body of a ``/query`` or ``/retrieve`` response: the answer and its records.

What a client of a workflow app reads: the answer, the scored vector-store
records as their store holds them, under ``records`` (ADR 0014), and the time
and the tokens of each stage. The mode and the name of the total timing come
with it: the workflow's result and the load-test client name them too.

Example:
    >>> QueryResponse.model_validate_json(body).records[0].id
    'doc-0001:0'
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from query_processing_core.retrieval.metadata_filter import MetadataCondition
from query_processing_core.retrieval.vector_store_record import (
    ScoredVectorStoreRecord,
)

RagMode = Literal["query", "retrieve"]
# The key of ``timings_ms`` that holds the whole run's wall time. Every other
# key is a stage of the graph that ran, so no list of stages is kept here: the
# workflow and the load-test client share only this name of the wire.
TOTAL_TIMING_KEY = "total"


class QueryResponse(BaseModel):
    """The workflow's answer, its records, the time and tokens of each stage.

    Each record is returned as it is: a text, the flat metadata with its id
    and, for a record made from a chunk, the chunk, and the two scores.
    ``tokens`` holds, under each stage whose LLM reported both counts, the
    ``prompt`` and ``completion`` tokens; ``{}`` when no stage did, as on a
    ``/retrieve`` that asks the LLM nothing.

    Example:
        >>> QueryResponse(request_id="r", topology=None, run_label="fake",
        ...     question="q", mode="retrieve", answer=None, records=[],
        ...     timings_ms={TOTAL_TIMING_KEY: 1.5}, tokens={},
        ...     metadata_filter=[]).mode
        'retrieve'
    """

    request_id: str
    # The topology alias, as X-RAG-Topology; None when none was given.
    topology: str | None
    run_label: str
    question: str
    mode: RagMode
    answer: str | None
    records: list[ScoredVectorStoreRecord]
    timings_ms: dict[str, float]
    tokens: dict[str, dict[str, int]]
    # The conditions applied: the client's, then those the stages added.
    metadata_filter: list[MetadataCondition]
