"""The vector store ports: storing and finding passages by vector, and by words.

Each port is an abstract base class: an implementation states which port it
implements by subclassing it, and one that forgets a method fails when it is
instantiated, not on the first request that needs the method. Every method is
async so the event loop never blocks: in-process implementations offload to a
thread pool, HTTP ones await httpx. This module imports the domain only: the
application depends on it, and on nothing else of the adapters.

As a component, a vector store answers a ``SearchRequest`` by its retrieval
mode: ``dense`` asks it by vector, ``lexical`` by words, ranked by the store, and
``hybrid`` does both at once and fuses the two rankings by reciprocal rank, so a
hybrid search behaves the same on every store. Lexical and hybrid need a store
that ranks text: one that is also a ``BaseLexicalSearch``.

Example:
    >>> isinstance(store, BaseLexicalSearch)  # only a store that ranks text
    True
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence

from ...domain.base_component import BaseComponent
from ...domain.component_requests import SearchAnswer, SearchRequest
from ...domain.metadata_filter import MetadataCondition
from ...domain.passages import Passage, ScoredPassage
from ...domain.rank_fusion import reciprocal_rank_fusion


class BaseVectorStore(
    BaseComponent[SearchRequest, SearchAnswer], component_type="vector_store"
):
    """Stores passages with their vectors and finds them by vector."""

    async def ainvoke(self, request: SearchRequest) -> SearchAnswer:
        """Search the store by the request's retrieval mode.

        Args:
            request: The vector, the words or both, how many passages to return,
                the retrieval mode and, for ``hybrid``, the fusion's rank constant.

        Returns:
            At most ``top_k`` passages, best first.

        Raises:
            ValueError: The retrieval mode is ``lexical`` or ``hybrid`` and the
                store ranks no text.

        Example:
            >>> await store.ainvoke(SearchRequest(
            ...     embedding=vector, top_k=20, retrieval_mode="dense"))
            SearchAnswer(passages=[...])
        """
        if request.retrieval_mode == "dense":
            return SearchAnswer(passages=await self._dense(request))
        search = required_lexical_search(self, request.retrieval_mode)
        lexical = search.lexical_query(
            request.text or "", request.top_k, request.metadata_filter
        )
        if request.retrieval_mode == "lexical":
            return SearchAnswer(passages=await lexical)
        # Both legs at once, each asked for top_k, fused back to top_k. A hybrid
        # request always carries rrf_k: SearchRequest refuses one without it.
        rankings = await asyncio.gather(self._dense(request), lexical)
        fused = reciprocal_rank_fusion(
            rankings, k=request.rrf_k or 0, top_k=request.top_k
        )
        return SearchAnswer(passages=fused)

    async def _dense(self, request: SearchRequest) -> list[ScoredPassage]:
        return await self.query(
            request.embedding or [], request.top_k, request.metadata_filter
        )

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
        self,
        embedding: Sequence[float],
        top_k: int,
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> list[ScoredPassage]:
        """Find the passages nearest to a vector among those the filter keeps.

        Args:
            embedding: The query vector.
            top_k: How many passages to return at most.
            metadata_filter: Conditions a passage must all meet; none filters
                nothing.

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
    async def lexical_query(
        self,
        text: str,
        top_k: int,
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> list[ScoredPassage]:
        """Find the passages whose words best match a text, among those kept.

        Args:
            text: The text searched for.
            top_k: How many passages to return at most.
            metadata_filter: Conditions a passage must all meet; none filters
                nothing.

        Returns:
            The best-matching passages, best first, ranked by the store's own
            text ranking, with its score as ``retrieval_score``; a passage
            sharing no word with ``text`` is left out.
        """


def required_lexical_search(store: BaseVectorStore, mode: str) -> BaseLexicalSearch:
    """Return the store's lexical search, which a lexical or hybrid retrieval needs.

    Args:
        store: The vector store searched.
        mode: The retrieval mode that needs it, named in the error.

    Returns:
        ``store``, known to rank text.

    Raises:
        ValueError: The store ranks no text.

    Example:
        >>> required_lexical_search(elasticsearch_store, "hybrid")
    """
    if isinstance(store, BaseLexicalSearch):
        return store
    raise ValueError(
        f"retrieval_mode={mode!r} needs a store that ranks text, but the vector "
        "store ranks no text: set the backend of the vector store it calls to "
        "elasticsearch (QP_COMPONENTS__VECTOR_STORE__BACKEND for the one named "
        "vector_store), or the retrieve stage's retrieval_mode to dense"
    )
