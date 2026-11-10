"""Value objects, ports and errors shared by every module.

Nothing heavy is imported here, so the OVMS-only and fake topologies never pay
for torch, elasticsearch or openai.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

MetadataValue = str | int | float | bool
RagMode = Literal["query", "retrieve"]
# Workflow stages, in execution order; also the keys of ``RagResult.timings_ms``.
STAGES: tuple[str, ...] = ("embed", "retrieve", "rerank", "generate")


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
    """Outcome of one workflow run; ``timings_ms`` has one key per stage + ``total``."""

    question: str
    mode: RagMode
    answer: str | None
    passages: list[ScoredPassage]
    timings_ms: dict[str, float]


def top_by_score(
    passages: Sequence[ScoredPassage], scores: Sequence[float], top_n: int
) -> list[ScoredPassage]:
    """Keep the ``top_n`` passages with the highest score, stamped as ``rerank_score``.

    Example::

        top_by_score([a, b, c], [0.1, 0.9, 0.5], 2)  # [b(0.9), c(0.5)]
    """
    ranked = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
    return [
        passages[i].model_copy(update={"rerank_score": float(scores[i])})
        for i in ranked[:top_n]
    ]


# --- ports -----------------------------------------------------------------
# Every method is async so the FastAPI event loop never blocks: in-process
# adapters offload to a thread pool, HTTP adapters await httpx.


@runtime_checkable
class EmbedderPort(Protocol):
    model_name: str

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def ready(self) -> tuple[bool, str]: ...


@runtime_checkable
class RerankerPort(Protocol):
    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]: ...

    async def ready(self) -> tuple[bool, str]: ...


@runtime_checkable
class VectorStorePort(Protocol):
    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None: ...

    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]: ...

    async def count(self) -> int: ...


@runtime_checkable
class ChatModelPort(Protocol):
    model_name: str

    async def generate(self, messages: list[dict[str, str]]) -> str: ...


# --- errors ----------------------------------------------------------------


class RagDependencyError(RuntimeError):
    """A dependency failed; ``kind`` is ``unavailable`` (HTTP 503) or ``timeout`` (504).

    Example::

        raise RagDependencyError("reranker", "HTTP 503", target=url)
    """

    def __init__(
        self,
        component: str,
        detail: str,
        *,
        target: str | None = None,
        kind: str = "unavailable",
    ) -> None:
        self.component, self.detail, self.target, self.kind = (
            component,
            detail,
            target,
            kind,
        )
        suffix = f" (target={target})" if target else ""
        super().__init__(f"{component} {kind}: {detail}{suffix}")
