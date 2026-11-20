"""The retrieve stage type: candidates from the vector store, by its retrieval mode.

The stage builds a ``SearchRequest`` from the fields its ``reads`` names, in
order: the query's vector for ``dense``, its words for ``lexical``, both for
``hybrid``, then the metadata filter every record found must meet. Its
retrieval mode and ``top_k`` complete the request. The store's ``ainvoke``
searches by that mode, and fuses a hybrid search; the records found go
to the one field its ``writes`` names. Lexical and hybrid need a store that
ranks text: the store's own ``BaseLexicalSearch``, refused when the stage is
built if it has none.

Example:
    >>> stage = RetrieveStage(
    ...     store, RetrieveSettings(retrieval_mode="hybrid", top_k=20),
    ...     StageFields(("query_embedding", "standalone_query", "metadata_filter"),
    ...                 ("candidates",)),
    ...     StageContext())
    >>> await stage.run({"query_embedding": vector, "standalone_query": "vacation",
    ...                  "metadata_filter": []})
    {'candidates': [...]}
"""

from __future__ import annotations

from typing import Any

from pydantic import Field
from query_processing_core.retrieval.metadata_filter import MetadataCondition
from query_processing_core.retrieval.vector_store_record import ScoredVectorStoreRecord

from ...adapters.vector_store.base_vector_store import (
    BaseVectorStore,
    required_lexical_search,
)
from ...domain.components.component_requests import SearchRequest
from ...domain.retrieval.retrieval_mode import RetrievalMode
from ..workflow_state import WorkflowState
from .base_stage import (
    STAGE_TYPES,
    BaseStage,
    FieldTypes,
    StageContext,
    StageFields,
    StageSettings,
    fields_in_order,
)

# What every retrieval mode reads last: the conditions a record must meet.
_FILTER_READ = {"metadata_filter": list[MetadataCondition]}
# What each retrieval mode reads of the state, in order: the search request's
# attributes its reads fill, each with its type.
READS_BY_RETRIEVAL_MODE: dict[RetrievalMode, dict[str, Any]] = {
    "dense": {"embedding": list[float], **_FILTER_READ},
    "lexical": {"text": str, **_FILTER_READ},
    "hybrid": {"embedding": list[float], "text": str, **_FILTER_READ},
}
# What it writes: the vector-store records found.
RETRIEVE_WRITE_TYPES = {"candidates": list[ScoredVectorStoreRecord]}


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
        fields: What its retrieval mode reads, and the one field it writes.
        context: What every stage is given; a retrieve stage reads none of it.

    Raises:
        ValueError: The lists do not fit the retrieval mode, or the retrieval
            mode is ``lexical`` or ``hybrid`` and the store ranks no text.
    """

    component_type = BaseVectorStore
    settings_type = RetrieveSettings

    def __init__(
        self,
        component: BaseVectorStore,
        settings: RetrieveSettings,
        fields: StageFields,
        context: StageContext,
    ) -> None:
        super().__init__(component, settings, fields, context)
        # Checked here as well as in the store's ainvoke, so a lexical or hybrid
        # workflow over a store that ranks no text fails at start-up, not on its
        # first request.
        if settings.retrieval_mode != "dense":
            required_lexical_search(component, settings.retrieval_mode)

    @classmethod
    def check_fields(
        cls, settings: RetrieveSettings, fields: StageFields, context: StageContext
    ) -> FieldTypes:
        """Return the types of what a retrieve stage reads and writes by its mode.

        Args:
            settings: The stage's settings; only the retrieval mode is read.
            fields: The query's vector, its words, or both, in that order, then
                the metadata filter; and the one field the candidates go to.
            context: Not read: a retrieve stage names no prompt.

        Returns:
            Each field's type.

        Raises:
            ValueError: ``reads`` does not list what the retrieval mode needs,
                or ``writes`` does not list one field.

        Example:
            >>> RetrieveStage.check_fields(
            ...     RetrieveSettings(retrieval_mode="lexical", top_k=20),
            ...     StageFields(("standalone_query", "metadata_filter"),
            ...                 ("candidates",)),
            ...     StageContext()).reads
            {'standalone_query': <class 'str'>, 'metadata_filter': ...}
        """
        del context  # a retrieve stage names no prompt
        mode = settings.retrieval_mode
        expecting = f"the retrieve stage in retrieval_mode {mode!r}"
        reads = READS_BY_RETRIEVAL_MODE[mode]
        return FieldTypes(
            fields_in_order("reads", fields.reads, reads, expecting),
            fields_in_order("writes", fields.writes, RETRIEVE_WRITE_TYPES, expecting),
        )

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Search the store.

        Args:
            state: The workflow state; the fields of ``reads`` are read.

        Returns:
            The field of ``writes``: at most ``top_k`` records, best first.
        """
        attributes = READS_BY_RETRIEVAL_MODE[self.settings.retrieval_mode]
        request = SearchRequest(
            **dict(zip(attributes, self.read_fields(state), strict=True)),
            top_k=self.settings.top_k,
            retrieval_mode=self.settings.retrieval_mode,
            rrf_k=self.settings.rrf_k,
        )
        return self.write_field((await self.component.ainvoke(request)).records)
