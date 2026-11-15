"""The retrieve stage type: candidates from the vector store, by its retrieval mode.

``dense`` asks the store by the query's vector, ``lexical`` by the words of the
standalone query, and ``hybrid`` asks both at once and fuses the two rankings by
reciprocal rank. Lexical and hybrid need a store that ranks text: the store's
own ``BaseLexicalSearch``, refused when the stage is built if it has none.

Example:
    >>> stage = RetrieveStage(store, RetrieveSettings(
    ...     retrieval_mode="hybrid", top_k=20, rrf_k=60))
    >>> await stage.run({"query_embedding": vector, "standalone_query": "vacation"})
    {'candidates': [...]}
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from pydantic import Field

from ...adapters.vector_store.base_vector_store import (
    BaseLexicalSearch,
    BaseVectorStore,
)
from ...domain.passages import ScoredPassage
from ...domain.rank_fusion import reciprocal_rank_fusion
from ...domain.retrieval_mode import RetrievalMode
from ..workflow_state import WorkflowState
from .base_stage import STAGE_TYPES, BaseStage, StageSettings

# What each retrieval mode reads of the state: a vector, words, or both.
READS_BY_RETRIEVAL_MODE: dict[RetrievalMode, frozenset[str]] = {
    "dense": frozenset({"query_embedding"}),
    "lexical": frozenset({"standalone_query"}),
    "hybrid": frozenset({"query_embedding", "standalone_query"}),
}

# The search of one retrieval mode: state in, candidates out.
CandidateFinder = Callable[[WorkflowState], Awaitable[list[ScoredPassage]]]


class RetrieveSettings(StageSettings):
    """How the retrieve stage searches, how many candidates it keeps, and the fusion.

    ``rrf_k`` is the rank constant of the fusion a ``hybrid`` retrieval ends
    with.
    """

    retrieval_mode: RetrievalMode
    top_k: int = Field(ge=1)
    rrf_k: int = Field(ge=1)


def required_lexical_search(store: BaseVectorStore, mode: str) -> BaseLexicalSearch:
    """Return the store's lexical search, which a lexical or hybrid retrieval needs.

    Args:
        store: The vector store the retrieve stage calls.
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
        "store ranks no text: set RAG_VECTOR_STORE_BACKEND=elasticsearch, or "
        "RAG_RETRIEVAL_MODE=dense"
    )


@STAGE_TYPES.register("retrieve")
class RetrieveStage(BaseStage[BaseVectorStore, RetrieveSettings]):
    """Find ``top_k`` candidates in the vector store by the retrieval mode.

    Args:
        component: The vector store; a store that ranks text for a lexical or
            hybrid retrieval.
        settings: The retrieval mode, the number of candidates and the fusion's
            rank constant.

    Raises:
        ValueError: The retrieval mode is ``lexical`` or ``hybrid`` and the
            store ranks no text.
    """

    component_type = BaseVectorStore
    settings_type = RetrieveSettings
    writes = frozenset({"candidates"})

    def __init__(self, component: BaseVectorStore, settings: RetrieveSettings) -> None:
        super().__init__(component, settings)
        self.reads = READS_BY_RETRIEVAL_MODE[settings.retrieval_mode]
        self._find_candidates = self._candidate_finder()

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Search the store.

        Args:
            state: The workflow state; the fields of ``reads`` are read.

        Returns:
            ``candidates``: at most ``top_k`` passages, best first.
        """
        return {"candidates": await self._find_candidates(state)}

    # Chosen once at construction, so a lexical or hybrid workflow over a store
    # that ranks no text fails at start-up, not on its first request.
    def _candidate_finder(self) -> CandidateFinder:
        store, top_k = self.component, self.settings.top_k
        mode, rrf_k = self.settings.retrieval_mode, self.settings.rrf_k

        async def dense(state: WorkflowState) -> list[ScoredPassage]:
            return await store.query(state["query_embedding"], top_k)

        if mode == "dense":
            return dense
        search = required_lexical_search(store, mode)

        async def lexical(state: WorkflowState) -> list[ScoredPassage]:
            return await search.lexical_query(state["standalone_query"], top_k)

        if mode == "lexical":
            return lexical

        async def hybrid(state: WorkflowState) -> list[ScoredPassage]:
            # Both legs at once, each asked for top_k, fused back to top_k.
            rankings = await asyncio.gather(dense(state), lexical(state))
            return reciprocal_rank_fusion(rankings, k=rrf_k, top_k=top_k)

        return hybrid
