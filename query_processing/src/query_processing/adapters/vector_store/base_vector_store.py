"""The vector store ports: storing and finding records by vector, and by words.

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

The contract of every store on a record's id (ADR 0014), which a record keeps in
its metadata, under ``id``: on a write, the record's id is the store's id for it,
and its metadata is stored as it is, the id with it; a record read has the id
the store holds it under, whatever its stored metadata holds, so an index
ingested before, whose stored metadata holds no id, still answers.
``record_as_read`` builds a record read that way.

Example:
    >>> isinstance(store, BaseLexicalSearch)  # only a store that ranks text
    True
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from query_processing_core.retrieval.metadata_filter import (
    MetadataCondition,
    MetadataValue,
)
from query_processing_core.retrieval.vector_store_record import (
    ScoredVectorStoreRecord,
    VectorStoreRecord,
)

from ...domain.components.base_component import BaseComponent
from ...domain.components.component_requests import SearchAnswer, SearchRequest
from ...domain.retrieval.rank_fusion import reciprocal_rank_fusion


class BaseVectorStore(
    BaseComponent[SearchRequest, SearchAnswer], component_type="vector_store"
):
    """Stores records with their vectors and finds them by vector.

    ``prefilter`` holds the conditions the store adds to every search, in front
    of a request's own: which records a workflow can read at all. Its opener
    sets it from the store's settings; none by default.
    """

    prefilter: Sequence[MetadataCondition] = ()

    async def ainvoke(self, request: SearchRequest) -> SearchAnswer:
        """Search the store by the request's retrieval mode, under its prefilter.

        Args:
            request: The vector, the words or both, how many records to return,
                the retrieval mode, the metadata filter and, for ``hybrid``, the
                fusion's rank constant.

        Returns:
            At most ``top_k`` records, best first, each meeting the prefilter
            and the request's metadata filter.

        Raises:
            ValueError: The retrieval mode is ``lexical`` or ``hybrid`` and the
                store ranks no text.

        Example:
            >>> await store.ainvoke(SearchRequest(
            ...     embedding=vector, top_k=20, retrieval_mode="dense"))
            SearchAnswer(records=[...])
        """
        conditions = [*self.prefilter, *request.metadata_filter]
        if request.retrieval_mode == "dense":
            return SearchAnswer(records=await self._dense(request, conditions))
        search = required_lexical_search(self, request.retrieval_mode)
        lexical = search.lexical_query(request.text or "", request.top_k, conditions)
        if request.retrieval_mode == "lexical":
            return SearchAnswer(records=await lexical)
        # Both legs at once, each asked for top_k, fused back to top_k. A hybrid
        # request always carries rrf_k: SearchRequest refuses one without it.
        rankings = await asyncio.gather(self._dense(request, conditions), lexical)
        fused = reciprocal_rank_fusion(
            rankings, k=request.rrf_k or 0, top_k=request.top_k
        )
        return SearchAnswer(records=fused)

    async def _dense(
        self, request: SearchRequest, conditions: Sequence[MetadataCondition]
    ) -> list[ScoredVectorStoreRecord]:
        return await self.query(request.embedding or [], request.top_k, conditions)

    @abstractmethod
    async def upsert(
        self,
        records: Sequence[VectorStoreRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Store records with their vectors, each under its record's id.

        The metadata is stored as it is, the id with it.

        Args:
            records: The records to store, each with an ``id`` in its metadata.
            embeddings: One vector per record, in the same order.
        """

    @abstractmethod
    async def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> list[ScoredVectorStoreRecord]:
        """Find the records nearest to a vector among those the filter keeps.

        Args:
            embedding: The query vector.
            top_k: How many records to return at most.
            metadata_filter: Conditions a record must all meet; none filters
                nothing.

        Returns:
            The nearest records, nearest first, with their ``retrieval_score``,
            each with the id the store holds it under (``record_as_read``).
        """

    @abstractmethod
    async def count(self) -> int:
        """Count the stored records.

        Returns:
            The number of records in the store.
        """

    @abstractmethod
    async def ready(self) -> tuple[bool, str]:
        """Report whether the store can serve requests.

        Returns:
            Whether it is ready, and a short human-readable status such as how many
            records it holds.
        """


class BaseLexicalSearch(ABC):
    """Ranks stored records by how well their words match a text.

    A port of its own, apart from ``BaseVectorStore``, because not every store
    ranks text: open-source Chroma does not, Elasticsearch does.
    """

    @abstractmethod
    async def lexical_query(
        self,
        text: str,
        top_k: int,
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> list[ScoredVectorStoreRecord]:
        """Find the records whose words best match a text, among those kept.

        Args:
            text: The text searched for.
            top_k: How many records to return at most.
            metadata_filter: Conditions a record must all meet; none filters
                nothing.

        Returns:
            The best-matching records, best first, ranked by the store's own
            text ranking, with its score as ``retrieval_score``; a record
            sharing no word with ``text`` is left out. Each has the id the
            store holds it under (``record_as_read``).
        """


def record_as_read(
    store_id: str,
    text: str,
    metadata: Mapping[str, MetadataValue],
    retrieval_score: float,
) -> ScoredVectorStoreRecord:
    """Build a record found by a search, with the id its store holds it under.

    The store's id overrides any ``id`` the stored metadata holds, and stands in
    for one it lacks, as in an index ingested before records kept their id in
    their metadata.

    Args:
        store_id: The id the store holds the record under.
        text: The record's stored text.
        metadata: Its stored metadata, with or without an ``id``.
        retrieval_score: The score the search gave it.

    Returns:
        The scored record, its ``id`` the store's.

    Example:
        >>> record_as_read("p1", "t", {"topic": "hr"}, 0.5).metadata
        {'topic': 'hr', 'id': 'p1'}
    """
    return ScoredVectorStoreRecord(
        text=text,
        metadata={**metadata, "id": store_id},
        retrieval_score=float(retrieval_score),
    )


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
