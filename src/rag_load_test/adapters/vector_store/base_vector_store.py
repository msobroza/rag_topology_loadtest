"""The vector store ports: storing and finding passages by vector, and by words.

Each port is an abstract base class: an implementation states which port it
implements by subclassing it, and one that forgets a method fails when it is
instantiated, not on the first request that needs the method. Every method is
async so the event loop never blocks: in-process implementations offload to a
thread pool, HTTP ones await httpx. This module imports the domain only: the
application depends on it, and on nothing else of the adapters.

Example:
    >>> isinstance(store, BaseLexicalSearch)  # only a store that ranks text
    True
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ...domain.passages import Passage, ScoredPassage


class BaseVectorStore(ABC):
    """Stores passages with their vectors and finds them by vector."""

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


class BaseLexicalSearch(ABC):
    """Ranks stored passages by how well their words match a text.

    A port of its own, apart from ``BaseVectorStore``, because not every store
    ranks text: open-source Chroma does not, Elasticsearch does.
    """

    @abstractmethod
    async def lexical_query(self, text: str, top_k: int) -> list[ScoredPassage]:
        """Find the passages whose words best match a text.

        Args:
            text: The text searched for.
            top_k: How many passages to return at most.

        Returns:
            The best-matching passages, best first, ranked by the store's own
            text ranking, with its score as ``retrieval_score``; a passage
            sharing no word with ``text`` is left out.
        """
