"""The four async ports a workflow run needs, and their bundle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .passages import Passage, ScoredPassage

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

    async def ready(self) -> tuple[bool, str]: ...


@runtime_checkable
class ChatModelPort(Protocol):
    model_name: str

    async def generate(self, messages: list[dict[str, str]]) -> str: ...


@dataclass
class RagDependencies:
    """One open port of each kind: everything a workflow run needs."""

    embedder: EmbedderPort
    vector_store: VectorStorePort
    reranker: RerankerPort
    chat_model: ChatModelPort
