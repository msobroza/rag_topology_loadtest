"""The four async ports a workflow run needs, and their bundle.

Each port is an abstract base class: an adapter states which port it implements
by subclassing it, and one that forgets a method fails when it is instantiated,
not on the first request that needs the method.

Example:
    >>> class MyReranker(RerankerPort):
    ...     async def rerank(self, query, passages, top_n): ...
    ...     async def ready(self): return True, "mine"
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
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text, in the same order.
        """

    @abstractmethod
    async def ready(self) -> tuple[bool, str]:
        """Report whether the embedder can serve requests.

        Returns:
            Whether it is ready, and a short human-readable status such as the
            model name.
        """


class RerankerPort(ABC):
    """Re-scores retrieval candidates against the query and keeps the best."""

    @abstractmethod
    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        """Re-score retrieval candidates against the query and keep the best.

        Args:
            query: The question the passages are scored against.
            passages: The retrieval candidates.
            top_n: How many passages to keep.

        Returns:
            At most ``top_n`` passages, best first.
        """

    @abstractmethod
    async def ready(self) -> tuple[bool, str]:
        """Report whether the reranker can serve requests.

        Returns:
            Whether it is ready, and a short human-readable status.
        """


class VectorStorePort(ABC):
    """Stores passages with their vectors and finds them by vector or by words."""

    @abstractmethod
    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Store passages with their vectors.

        Args:
            passages: The passages to store.
            embeddings: One vector per passage, in the same order.
        """

    @abstractmethod
    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]:
        """Find the passages nearest to a vector.

        Args:
            embedding: The query vector.
            top_k: How many passages to return at most.

        Returns:
            The nearest passages, nearest first, with their ``retrieval_score``.
        """

    @abstractmethod
    async def lexical_query(self, text: str, top_k: int) -> list[ScoredPassage]:
        """Find the passages whose words best match a text (BM25).

        Args:
            text: The text searched for.
            top_k: How many passages to return at most.

        Returns:
            The best-matching passages, best first, with the lexical score as
            ``retrieval_score``; a passage sharing no word with ``text`` is left
            out.
        """

    @abstractmethod
    async def count(self) -> int:
        """Count the stored passages.

        Returns:
            The number of passages in the store.
        """

    @abstractmethod
    async def ready(self) -> tuple[bool, str]:
        """Report whether the store can serve requests.

        Returns:
            Whether it is ready, and a short human-readable status such as how many
            passages it holds.
        """


class ChatModelPort(ABC):
    """Answers an OpenAI-style message list."""

    model_name: str

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Answer a conversation.

        Args:
            messages: OpenAI-style messages, each with ``role`` and ``content``.

        Returns:
            The model's answer text.
        """


@dataclass
class RagDependencies:
    """One open port of each kind: everything a workflow run needs."""

    embedder: EmbedderPort
    vector_store: VectorStorePort
    reranker: RerankerPort
    chat_model: ChatModelPort
