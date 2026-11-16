"""The retrieve stage type: candidates from the vector store, by its retrieval mode.

The stage builds a ``SearchRequest`` from the state: the query's vector, the
words of the standalone query, its retrieval mode and its ``top_k``. The store's
``ainvoke`` searches by that mode, and fuses a hybrid search. Lexical and hybrid
need a store that ranks text: the store's own ``BaseLexicalSearch``, refused when
the stage is built if it has none.

Example:
    >>> stage = RetrieveStage(
    ...     store, RetrieveSettings(retrieval_mode="hybrid", top_k=20))
    >>> await stage.run({"query_embedding": vector, "standalone_query": "vacation"})
    {'candidates': [...]}
"""

from __future__ import annotations

from pydantic import Field

from ...adapters.vector_store.base_vector_store import (
    BaseVectorStore,
    required_lexical_search,
)
from ...domain.component_requests import SearchRequest
from ...domain.retrieval_mode import RetrievalMode
from ..workflow_state import WorkflowState
from .base_stage import STAGE_TYPES, BaseStage, StageSettings

# What each retrieval mode reads of the state: a vector, words, or both.
READS_BY_RETRIEVAL_MODE: dict[RetrievalMode, frozenset[str]] = {
    "dense": frozenset({"query_embedding"}),
    "lexical": frozenset({"standalone_query"}),
    "hybrid": frozenset({"query_embedding", "standalone_query"}),
}


class RetrieveSettings(StageSettings):
    """How the retrieve stage searches, how many candidates it keeps, and the fusion.

    The retrieval mode and ``top_k`` are the use case's choices, so a workflow
    file states them. ``rrf_k`` is the rank constant of the fusion a ``hybrid``
    retrieval ends with, a technical default a file may leave out.
    """

    retrieval_mode: RetrievalMode
    top_k: int = Field(ge=1)
    # 60, as in the paper that defined the fusion (Cormack, Clarke and Buettcher,
    # SIGIR 2009); smaller widens the gap between ranks.
    rrf_k: int = Field(default=60, ge=1)


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
        self.reads = self.fields_read(settings)
        # Checked here as well as in the store's ainvoke, so a lexical or hybrid
        # workflow over a store that ranks no text fails at start-up, not on its
        # first request.
        if settings.retrieval_mode != "dense":
            required_lexical_search(component, settings.retrieval_mode)

    @classmethod
    def fields_read(cls, settings: RetrieveSettings) -> frozenset[str]:
        """Return what a retrieve stage reads: what its retrieval mode needs.

        Args:
            settings: The stage's settings; only the retrieval mode is read.

        Returns:
            The query's vector, its words, or both.

        Example:
            >>> RetrieveStage.fields_read(RetrieveSettings(retrieval_mode="dense",
            ...                                            top_k=20))
            frozenset({'query_embedding'})
        """
        return READS_BY_RETRIEVAL_MODE[settings.retrieval_mode]

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Search the store.

        Args:
            state: The workflow state; the fields of ``reads`` are read.

        Returns:
            ``candidates``: at most ``top_k`` passages, best first.
        """
        request = SearchRequest(
            embedding=state.get("query_embedding"),
            text=state.get("standalone_query"),
            top_k=self.settings.top_k,
            retrieval_mode=self.settings.retrieval_mode,
            rrf_k=self.settings.rrf_k,
        )
        return {"candidates": (await self.component.ainvoke(request)).passages}
