"""The generate stage type: a prompt rendered from the fields it reads, the answer out.

A generate stage names its prompt, one of the prompts file's (spec 4). The
fields its ``reads`` names are the prompt's variables, as many as it uses, by
their names; the LLM's answer goes to the one field its ``writes`` names.

Example:
    >>> stage = GenerateStage(llm, GenerateSettings(prompt="answer_from_context"),
    ...                       StageFields(("conversational_query", "reranked"),
    ...                                   ("answer",)),
    ...                       StageContext(settings.prompts))
    >>> await stage.run({"conversational_query": "vacation days?", "reranked": kept})
    {'answer': '...'}
"""

from __future__ import annotations

from typing import Any

from ...adapters.llm.base_llm import BaseLLM
from ...domain.prompts import Prompt
from ..prompt_rendering import prompt_variables, render_prompt
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

# What the stage writes: the answer's text.
GENERATE_WRITE_TYPES = {"answer": str}


class GenerateSettings(StageSettings):
    """The name of the prompt the stage renders, one of the prompts file's."""

    prompt: str


@STAGE_TYPES.register("generate")
class GenerateStage(BaseStage[BaseLLM, GenerateSettings]):
    """Ask the LLM the prompt its settings name, rendered from the fields it reads.

    Args:
        component: The LLM.
        settings: The name of its prompt.
        fields: The fields its prompt reads, and the one field the answer goes to.
        context: The prompts, by name.
    """

    component_type = BaseLLM
    settings_type = GenerateSettings

    def __init__(
        self,
        component: BaseLLM,
        settings: GenerateSettings,
        fields: StageFields,
        context: StageContext,
    ) -> None:
        super().__init__(component, settings, fields, context)
        self._prompt = context.prompts[settings.prompt]  # check_fields found it

    @classmethod
    def check_fields(
        cls, settings: GenerateSettings, fields: StageFields, context: StageContext
    ) -> FieldTypes:
        """Return the types of what a generate stage reads and writes.

        Args:
            settings: The name of its prompt.
            fields: Any number of fields, the prompt's variables; then the one
                field the answer goes to.
            context: The prompts, by name.

        Returns:
            ``Any`` for each field read, which a template takes as it is, and
            ``str`` for the one written.

        Raises:
            ValueError: The prompt is not defined, a template cannot be read
                (the message names its message), the prompt uses a variable
                the stage does not read, or ``writes`` does not list one field.

        Example:
            >>> GenerateStage.check_fields(
            ...     GenerateSettings(prompt="answer_from_context"),
            ...     StageFields(("conversational_query", "reranked"), ("answer",)),
            ...     StageContext(settings.prompts)).writes
            {'answer': <class 'str'>}
        """
        prompt = _named_prompt(settings.prompt, context)
        unread = sorted(prompt_variables(prompt) - set(fields.reads))
        if unread:
            raise ValueError(
                f"the prompt {settings.prompt!r} uses {', '.join(map(repr, unread))}, "
                f"which the stage does not read (it reads {list(fields.reads)}; "
                "expected each variable of the prompt among its reads)"
            )
        expecting = "the generate stage"
        return FieldTypes(
            dict.fromkeys(fields.reads, Any),
            fields_in_order("writes", fields.writes, GENERATE_WRITE_TYPES, expecting),
        )

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Ask the LLM its prompt, rendered from the fields its ``reads`` names.

        Args:
            state: The workflow state; the fields of ``reads`` are read.

        Returns:
            The field of ``writes``: the answer.
        """
        variables = dict(zip(self.fields.reads, self.read_fields(state), strict=True))
        messages = render_prompt(self._prompt, variables)
        return self.write_field(await self.component.generate(messages))


def _named_prompt(name: str, context: StageContext) -> Prompt:
    # The prompt a stage names, refused with the prompts the file defines.
    if name in context.prompts:
        return context.prompts[name]
    known = ", ".join(map(repr, context.prompts)) or "none"
    raise ValueError(
        f"names the prompt {name!r}, which the prompts file does not define "
        f"(its prompts: {known})"
    )
