"""The four async ports a workflow run needs, and their bundle.

Each port is an abstract base class: an adapter states which port it implements
by subclassing it, and one that forgets a method fails when it is instantiated,
not on the first request that needs the method.

Example::

    class MyReranker(RerankerPort):
        async def rerank(self, query, passages, top_n): ...
        async def ready(self): return True, "mine"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from .passages import Passage, ScoredPassage

# --- ports -----------------------------------------------------------------
# Every method is async so the FastAPI event loop never blocks: in-process
# adapters offload to a thread pool, HTTP adapters await httpx.


class EmbedderPort(ABC):
    """Turns texts into vectors."""

    model_name: str

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    async def ready(self) -> tuple[bool, str]: ...


class RerankerPort(ABC):
    """Re-scores retrieval candidates against the query and keeps the best."""

    @abstractmethod
    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]: ...

    @abstractmethod
    async def ready(self) -> tuple[bool, str]: ...


class VectorStorePort(ABC):
    """Stores passages with their vectors and finds the nearest ones."""

    @abstractmethod
    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None: ...

    @abstractmethod
    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]: ...

    @abstractmethod
    async def count(self) -> int: ...

    @abstractmethod
    async def ready(self) -> tuple[bool, str]: ...


class ChatModelPort(ABC):
    """Answers an OpenAI-style message list."""

    model_name: str

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]]) -> str: ...


@dataclass
class RagDependencies:
    """One open port of each kind: everything a workflow run needs."""

    embedder: EmbedderPort
    vector_store: VectorStorePort
    reranker: RerankerPort
    chat_model: ChatModelPort
