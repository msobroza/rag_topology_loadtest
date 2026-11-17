"""Value objects of a RAG run: passages and results.

Nothing heavy is imported here, so the OVMS-only and fake topologies never pay
for torch or elasticsearch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from .metadata_filter import MetadataCondition, MetadataValue

RagMode = Literal["query", "retrieve"]
# The key of ``RagResult.timings_ms`` that holds the whole run's wall time. Every
# other key is a stage of the graph that ran, so no list of stages is kept here:
# the workflow and the load-test client share only this name of the wire.
TOTAL_TIMING_KEY = "total"


class Passage(BaseModel):
    """A chunk of text stored in the vector store."""

    id: str
    text: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class ScoredPassage(Passage):
    """A passage returned by retrieval, optionally re-scored by the reranker."""

    retrieval_score: float
    rerank_score: float | None = None


class RagResult(BaseModel):
    """Outcome of one workflow run.

    ``timings_ms`` has one key per stage of the graph, in the order of its
    description, then ``TOTAL_TIMING_KEY``. ``tokens`` has one key per stage
    whose LLM reported both counts in this run, in the same order, each
    ``{"prompt": n, "completion": m}``; empty when none did, so a reader never
    mistakes a missing count for 0.

    Example:
        >>> RagResult(question="q", mode="retrieve", answer=None, passages=[],
        ...           timings_ms={}).tokens
        {}
    """

    question: str
    mode: RagMode
    answer: str | None
    passages: list[ScoredPassage]
    timings_ms: dict[str, float]
    tokens: dict[str, dict[str, int]] = Field(default_factory=dict)
    # The conditions applied: the client's, then those the stages added.
    metadata_filter: list[MetadataCondition] = Field(default_factory=list)


def top_by_score(
    passages: Sequence[ScoredPassage], scores: Sequence[float], top_n: int
) -> list[ScoredPassage]:
    """Keep the ``top_n`` passages with the highest score.

    Args:
        passages: The candidates to rank.
        scores: One score per passage, in the same order.
        top_n: How many passages to keep.

    Returns:
        Copies of the best passages, highest score first, each with its score
        stamped as ``rerank_score``.

    Example:
        >>> top_by_score([a, b, c], [0.1, 0.9, 0.5], 2)
        [b(0.9), c(0.5)]
    """
    ranked = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
    return [
        passages[i].model_copy(update={"rerank_score": float(scores[i])})
        for i in ranked[:top_n]
    ]
