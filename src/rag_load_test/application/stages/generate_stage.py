"""The generate stage type: the user's message and the passages in, the answer out.

Example:
    >>> stage = GenerateStage(llm, GenerateSettings(system_prompt="Cite [n].",
    ...                       user_prompt="Context: {context} Q: {question}"))
    >>> await stage.run({"conversational_query": "vacation days?", "reranked": kept})
    {'answer': '...'}
"""

from __future__ import annotations

from pydantic import model_validator

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

    @model_validator(mode="after")
    def check_user_prompt(self) -> GenerateSettings:
        """Refuse a user template that ``RagPrompts`` cannot fill.

        Checked with the settings, so the start-up checks refuse it before any
        component opens, not only when the stage is built.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: The template lacks a placeholder or holds a brace it
                cannot fill; ``RagPrompts`` names it.
        """
        RagPrompts(system=self.system_prompt, user=self.user_prompt)
        return self


@STAGE_TYPES.register("generate")
class GenerateStage(BaseStage[BaseLLM, GenerateSettings]):
    """Answer what the user asked, from the reranked passages.

    Args:
        component: The LLM.
        settings: The two prompts, a user template its settings already
            checked.
    """

    component_type = BaseLLM
    settings_type = GenerateSettings
    reads = frozenset({"conversational_query", "reranked"})
    writes = frozenset({"answer"})

    def __init__(self, component: BaseLLM, settings: GenerateSettings) -> None:
        super().__init__(component, settings)
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
