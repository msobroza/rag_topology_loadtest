"""The rerank stage type: the candidates re-scored against the standalone query.

Example:
    >>> stage = RerankStage(reranker, RerankSettings(top_k=10))
    >>> await stage.run({"standalone_query": "vacation days?", "candidates": found})
    {'reranked': [...]}
"""

from __future__ import annotations

from pydantic import Field

from ...adapters.reranker.base_reranker import BaseReranker
from ..workflow_state import WorkflowState
from .base_stage import STAGE_TYPES, BaseStage, StageSettings


class RerankSettings(StageSettings):
    """How many passages the rerank stage keeps."""

    top_k: int = Field(ge=1)


@STAGE_TYPES.register("rerank")
class RerankStage(BaseStage[BaseReranker, RerankSettings]):
    """Keep the ``top_k`` candidates the reranker scores best."""

    component_type = BaseReranker
    settings_type = RerankSettings
    reads = frozenset({"standalone_query", "candidates"})
    writes = frozenset({"reranked"})

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Rerank ``candidates`` against ``standalone_query``.

        Args:
            state: The workflow state; its standalone query and candidates are
                read.

        Returns:
            ``reranked``: at most ``top_k`` passages, best first.
        """
        reranked = await self.component.rerank(
            state["standalone_query"], state["candidates"], self.settings.top_k
        )
        return {"reranked": reranked}
