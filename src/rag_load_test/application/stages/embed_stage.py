"""The embed stage type: the standalone query in, its vector out.

Example:
    >>> stage = EmbedStage(embedder, EmbedSettings())
    >>> await stage.run({"standalone_query": "vacation days?"})
    {'query_embedding': [...]}
"""

from __future__ import annotations

from ...adapters.embedder.base_embedder import BaseEmbedder
from ..workflow_state import WorkflowState
from .base_stage import STAGE_TYPES, BaseStage, StageSettings


class EmbedSettings(StageSettings):
    """The embed stage has no settings: the query prefix belongs to the embedder."""


@STAGE_TYPES.register("embed")
class EmbedStage(BaseStage[BaseEmbedder, EmbedSettings]):
    """Embed the standalone query as a query, for a dense or hybrid retrieval."""

    component_type = BaseEmbedder
    settings_type = EmbedSettings
    reads = frozenset({"standalone_query"})
    writes = frozenset({"query_embedding"})

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Embed ``standalone_query``.

        Args:
            state: The workflow state; only its standalone query is read.

        Returns:
            ``query_embedding``.
        """
        # A query, not a passage: the embedder puts its model's prefix in front.
        vector = await self.component.embed_query(state["standalone_query"])
        return {"query_embedding": vector}
