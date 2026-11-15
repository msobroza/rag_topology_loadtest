"""The generate stage type: the user's message and the passages in, the answer out.

Example:
    >>> stage = GenerateStage(llm, GenerateSettings(system_prompt="Cite [n].",
    ...                       user_prompt="Context: {context} Q: {question}"))
    >>> await stage.run({"conversational_query": "vacation days?", "reranked": kept})
    {'answer': '...'}
"""

from __future__ import annotations

from ...adapters.llm.base_llm import BaseLLM
from ...domain.prompts import RagPrompts
from ..workflow_state import WorkflowState
from .base_stage import STAGE_TYPES, BaseStage, StageSettings


class GenerateSettings(StageSettings):
    """The system prompt, and a user template holding ``{context}`` and ``{question}``.

    Any other literal brace of the template is doubled, as in ``str.format``.
    """

    system_prompt: str
    user_prompt: str


@STAGE_TYPES.register("generate")
class GenerateStage(BaseStage[BaseLLM, GenerateSettings]):
    """Answer what the user asked, from the reranked passages.

    Args:
        component: The LLM.
        settings: The two prompts.

    Raises:
        ValueError: The user template lacks a placeholder or holds a brace it
            cannot fill; ``RagPrompts`` names it.
    """

    component_type = BaseLLM
    settings_type = GenerateSettings
    reads = frozenset({"conversational_query", "reranked"})
    writes = frozenset({"answer"})

    def __init__(self, component: BaseLLM, settings: GenerateSettings) -> None:
        super().__init__(component, settings)
        # Built here, so a template RagPrompts refuses fails when the graph is
        # built, at start-up, not on the first query.
        self._prompts = RagPrompts(
            system=settings.system_prompt, user=settings.user_prompt
        )

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Ask the LLM about ``conversational_query`` with the ``reranked`` passages.

        Args:
            state: The workflow state; its conversational query and reranked
                passages are read.

        Returns:
            ``answer``.
        """
        # What the user asked, as asked: the standalone query is what is searched.
        messages = self._prompts.messages(
            state["conversational_query"], state["reranked"]
        )
        return {"answer": await self.component.generate(messages)}
