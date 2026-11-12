"""The reranker of ``RAG_RERANK_BACKEND=none``: no model, retrieval order kept.

It loads nothing and calls nothing, so it is not an in-process model adapter: it
sits at the root of the adapters, beside the module that picks it.

Example:
    >>> await NoReranker().rerank("q", passages, top_n=5)  # the first five
"""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.passages import ScoredPassage
from ..domain.ports import BaseReranker


class NoReranker(BaseReranker):
    """BaseReranker that keeps retrieval order (``RAG_RERANK_BACKEND=none``)."""

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        """Keep the first passages, in retrieval order.

        Args:
            query: Unused: retrieval order does not depend on it.
            passages: The candidates, best retrieval score first.
            top_n: How many to keep.

        Returns:
            The first ``top_n`` passages, as given.
        """
        del query  # the port's signature; retrieval order needs no query
        return list(passages[:top_n])

    async def ready(self) -> tuple[bool, str]:
        """Report ready; there is no model.

        Returns:
            ``True`` and ``"none"``.
        """
        return True, "none"
