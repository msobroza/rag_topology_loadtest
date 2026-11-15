"""The reranker port: re-scores retrieval candidates and keeps the best.

Each port is an abstract base class: an implementation states which port it
implements by subclassing it, and one that forgets a method fails when it is
instantiated, not on the first request that needs the method. Every method is
async so the event loop never blocks: in-process implementations offload to a
thread pool, HTTP ones await httpx. This module imports the domain only: the
application depends on it, and on nothing else of the adapters.

Example:
    >>> class MyReranker(BaseReranker):
    ...     async def rerank(self, query, passages, top_n): ...
    ...     async def ready(self): return True, "mine"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ...domain.passages import ScoredPassage


class BaseReranker(ABC):
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
