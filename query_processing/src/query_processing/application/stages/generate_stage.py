"""The generate stage type: the user's message and the passages in, the answer out.

Example:
    >>> stage = GenerateStage(llm, GenerateSettings(system_prompt="Cite [n].",
    ...                       user_prompt="Context: {context} Q: {question}"),
    ...                       StageFields(("conversational_query", "reranked"),
    ...                                   ("answer",)))
    >>> await stage.run({"conversational_query": "vacation days?", "reranked": kept})
    {'answer': '...'}
"""

from __future__ import annotations

from pydantic import model_validator

from ...adapters.llm.base_llm import BaseLLM
from ...domain.passages import ScoredPassage
from ...domain.prompts import RagPrompts
from ..workflow_state import WorkflowState
from .base_stage import (
    STAGE_TYPES,
    BaseStage,
    FieldTypes,
    StageFields,
    StageSettings,
    fields_in_order,
)

# What the stage reads, in order (the question asked, then the passages it is
# answered from), and what it writes (the answer's text), each with its type.
GENERATE_READ_TYPES = {"question": str, "passages": list[ScoredPassage]}
GENERATE_WRITE_TYPES = {"answer": str}


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
    """Answer what the user asked, from the passages kept.

    Args:
        component: The LLM.
        settings: The two prompts, a user template its settings already
            checked.
        fields: The question and the passages it reads, and the one field the
            answer goes to.
    """

    component_type = BaseLLM
    settings_type = GenerateSettings

    def __init__(
        self, component: BaseLLM, settings: GenerateSettings, fields: StageFields
    ) -> None:
        super().__init__(component, settings, fields)
        self._prompts = RagPrompts(
            system=settings.system_prompt, user=settings.user_prompt
        )

    @classmethod
    def check_fields(
        cls, settings: GenerateSettings, fields: StageFields
    ) -> FieldTypes:
        """Return the types of what a generate stage reads and writes.

        Args:
            settings: The stage's settings, which change nothing here.
            fields: The question and the passages, in that order; then the one
                field the answer goes to.

        Returns:
            Each field's type.

        Raises:
            ValueError: ``reads`` does not list two fields, or ``writes`` one.

        Example:
            >>> GenerateStage.check_fields(settings, StageFields(
            ...     ("conversational_query", "reranked"), ("answer",))).writes
            {'answer': <class 'str'>}
        """
        del settings  # the prompts do not change which fields it reads
        expecting = "the generate stage"
        return FieldTypes(
            fields_in_order("reads", fields.reads, GENERATE_READ_TYPES, expecting),
            fields_in_order("writes", fields.writes, GENERATE_WRITE_TYPES, expecting),
        )

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Ask the LLM the question with the passages its ``reads`` name.

        Args:
            state: The workflow state; the two fields of ``reads`` are read.

        Returns:
            The field of ``writes``: the answer.
        """
        question, passages = self.read_fields(state)
        messages = self._prompts.messages(question, passages)
        return self.write_field(await self.component.generate(messages))
