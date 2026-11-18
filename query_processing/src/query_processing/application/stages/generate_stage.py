"""The generate stage type: a prompt rendered from the fields it reads, the answer out.

A generate stage names its prompt, one of the prompts file's (spec 4). The
fields its ``reads`` names are the prompt's variables, as many as it uses, by
their names; the LLM's answer goes to the one field its ``writes`` names. A
stage that names an ``output_schema``, one of ``OUTPUT_SCHEMAS``, asks the LLM
for one instance of it and writes each of its fields to the field of
``writes`` of the same name (spec 5). A stage that names a vector store by
``metadata_fields_of`` asks for the schema narrowed to that store's declared
fields (``narrowed_to``), and its prompt also gets them as
``metadata_field_info``. A stage that states ``max_tokens`` caps its answer:
the LLM sends the cap to its endpoint. The cap is the stage's, not the LLM's,
so a rewrite under a schema is not cut short by the cap of an answer. The
stage hands the token counts its LLM reported to ``tokens``, which records
them under the stage's name.

Example:
    >>> stage = GenerateStage(llm, GenerateSettings(prompt="answer_from_context"),
    ...                       StageFields(("conversational_query", "reranked"),
    ...                                   ("answer",)),
    ...                       StageContext(settings.prompts))
    >>> await stage.run({"conversational_query": "vacation days?", "reranked": kept})
    {'answer': '...'}
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from ...adapters.llm.base_llm import BaseLLM
from ...domain.components.component_requests import GenerateAnswer
from ...domain.errors import RagDependencyError
from ...domain.prompts import Prompt
from ...domain.retrieval.metadata_filter import MetadataFieldInfo
from ..output_schemas import OUTPUT_SCHEMAS, condition_fields, narrowed_to
from ..prompt_rendering import prompt_placeholders, prompt_variables, render_prompt
from ..workflow_state import MESSAGES_TYPE, TOKENS_FIELD, WorkflowState
from .base_stage import (
    STAGE_TYPES,
    BaseStage,
    FieldTypes,
    StageContext,
    StageFields,
    StageSettings,
    fields_in_order,
)

# What the stage writes without an output schema: the answer's text.
GENERATE_WRITE_TYPES = {"answer": str}
# The prompt's variable that holds the fields metadata_fields_of names (spec 4).
METADATA_FIELD_INFO_VARIABLE = "metadata_field_info"


class GenerateSettings(StageSettings):
    """The prompt the stage renders, and the schema its answer follows, if any.

    ``prompt`` names one of the prompts file's prompts. ``output_schema`` names
    one of ``OUTPUT_SCHEMAS``; None asks the LLM for text.
    ``metadata_fields_of`` names a vector store: the schema's conditions are
    narrowed to the fields it declares, which the prompt also gets.
    ``max_tokens`` is the most tokens the answer may hold; None states no cap.
    """

    prompt: str
    output_schema: str | None = None
    metadata_fields_of: str | None = None
    max_tokens: int | None = None

    @field_validator("max_tokens")
    @classmethod
    def _a_cap_of_at_least_one(cls, cap: int | None) -> int | None:
        # A cap of 0 would ask for no answer; no cap is said by leaving it out.
        if cap is not None and cap < 1:
            raise ValueError(
                f"max_tokens is {cap} (expected a whole number of at least 1, "
                "or no max_tokens for no cap)"
            )
        return cap


@STAGE_TYPES.register("generate")
class GenerateStage(BaseStage[BaseLLM, GenerateSettings]):
    """Ask the LLM the prompt its settings name, rendered from the fields it reads.

    Args:
        component: The LLM.
        settings: The name of its prompt, and of its output schema if it has one.
        fields: The fields its prompt reads; then the one field the answer goes
            to, or under a schema the schema's fields, by name.
        context: The prompts, by name, and each vector store's declared fields.
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
        name = settings.output_schema
        self._schema = None if name is None else _named_schema(name)
        self._field_info = _declared_fields(settings, context)
        self._asked = _asked_schema(self._schema, self._field_info)

    @classmethod
    def check_fields(
        cls, settings: GenerateSettings, fields: StageFields, context: StageContext
    ) -> FieldTypes:
        """Return the types of what a generate stage reads and writes.

        Args:
            settings: The name of its prompt, and of its output schema if any.
            fields: Any number of fields, the prompt's variables; then the one
                field the answer goes to, or under a schema its fields.
            context: The prompts, by name, and each vector store's fields.

        Returns:
            ``Any`` for each field read, which a template takes as it is, or
            a list of messages for one a placeholder splices;
            ``str`` for the one written, or under a schema the type of each of
            its fields.

        Raises:
            ValueError: The prompt is not defined, a template cannot be read
                (the message names its message), the prompt uses a variable
                the stage does not read, the output schema is not registered,
                or ``writes`` does not list one field, or under a schema each
                of the schema's fields by name; ``metadata_fields_of`` names
                no vector store, or one that declares no field, or is missing
                under a schema that holds conditions.

        Example:
            >>> GenerateStage.check_fields(
            ...     GenerateSettings(prompt="answer_from_context"),
            ...     StageFields(("conversational_query", "reranked"), ("answer",)),
            ...     StageContext(settings.prompts)).writes
            {'answer': <class 'str'>}
        """
        prompt = _named_prompt(settings.prompt, context)
        field_info = _declared_fields(settings, context)
        given = set(fields.reads) | _extra_variables(field_info)
        unread = sorted(prompt_variables(prompt) - given)
        if unread:
            raise ValueError(
                f"the prompt {settings.prompt!r} uses {', '.join(map(repr, unread))}, "
                f"which the stage does not read (it reads {list(fields.reads)}; "
                "expected each variable of the prompt among its reads)"
            )
        if settings.output_schema is not None:  # narrowing fails at start-up
            schema = _named_schema(settings.output_schema)
            _refuse_undeclared_conditions(settings, schema, context)
            _asked_schema(schema, field_info)
        # A placeholder splices messages; a template takes any value as it is.
        spliced = prompt_placeholders(prompt)
        reads = {f: MESSAGES_TYPE if f in spliced else Any for f in fields.reads}
        return FieldTypes(reads, _write_types(settings, fields.writes))

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Ask the LLM its prompt, rendered from the fields its ``reads`` names.

        Args:
            state: The workflow state; the fields of ``reads`` are read.

        Returns:
            The field of ``writes``: the answer; under a schema, each of the
            schema's fields, from the instance the LLM answered; with
            ``answer_tokens`` of the answer, its token counts when it has both.

        Raises:
            RagDependencyError: Under a schema, the answer is not an instance
                of it; the message shows the answer.
        """
        variables = dict(zip(self.fields.reads, self.read_fields(state), strict=True))
        if self._field_info is not None:
            variables[METADATA_FIELD_INFO_VARIABLE] = list(self._field_info)
        messages = render_prompt(self._prompt, variables)
        cap = self.settings.max_tokens
        answered = await self.component.generate(messages, self._asked, cap)
        return {**self._written(answered.text), **answer_tokens(answered)}

    def _written(self, text: str) -> WorkflowState:
        # _asked is None only when _schema is: the second test is for the type checker.
        if self._schema is None or self._asked is None:
            return self.write_field(text)
        answer = _instance_of(self._asked, text)
        if self._asked is not self._schema:
            # Back to the schema's own types: the state holds the domain's
            # conditions. Dumped by alias, which the narrowed schema keeps.
            answer = self._schema.model_validate(answer.model_dump(by_alias=True))
        return {field: getattr(answer, field) for field in self.fields.writes}


def answer_tokens(answer: GenerateAnswer) -> WorkflowState:
    """Return the tokens an answer reports, unnamed, or nothing unless both counts are.

    One count alone is left out with the other: a tokens-a-second figure needs
    the completion count, and a prompt count alone would read as an answer of 0
    tokens. ``timed`` puts the counts under the stage's name.

    Args:
        answer: The LLM's answer, whose counts sum the attempts of one asked again.

    Returns:
        ``{"tokens": {"prompt": n, "completion": m}}``, or ``{}`` when either
        count is None.

    Example:
        >>> answer_tokens(GenerateAnswer(text="Paris.", prompt_tokens=829,
        ...                              completion_tokens=128))
        {'tokens': {'prompt': 829, 'completion': 128}}
    """
    prompt, completion = answer.prompt_tokens, answer.completion_tokens
    if prompt is None or completion is None:
        return {}
    return {TOKENS_FIELD: {"prompt": prompt, "completion": completion}}


def _named_prompt(name: str, context: StageContext) -> Prompt:
    # The prompt a stage names, refused with the prompts the file defines.
    if name in context.prompts:
        return context.prompts[name]
    known = ", ".join(map(repr, context.prompts)) or "none"
    raise ValueError(
        f"names the prompt {name!r}, which the prompts file does not define "
        f"(its prompts: {known})"
    )


def _write_types(settings: GenerateSettings, writes: tuple[str, ...]) -> dict[str, Any]:
    # The answer's text in one field, or each field of the schema by its name.
    if settings.output_schema is None:
        expecting = "the generate stage"
        return fields_in_order("writes", writes, GENERATE_WRITE_TYPES, expecting)
    name = settings.output_schema
    schema = _named_schema(name)
    schema_fields = list(schema.model_fields)
    if sorted(writes) != sorted(schema_fields):
        raise ValueError(
            f"the output schema {name!r} has the fields {schema_fields}, but the "
            f"stage writes {list(writes)} (expected writes listing each field of "
            "the schema, by name)"
        )
    return {field: schema.model_fields[field].annotation for field in writes}


def _declared_fields(
    settings: GenerateSettings, context: StageContext
) -> Sequence[MetadataFieldInfo] | None:
    # The fields of the vector store metadata_fields_of names; None without it.
    store = settings.metadata_fields_of
    if store is None:
        return None
    if store in context.metadata_field_info:
        return context.metadata_field_info[store]
    known = ", ".join(map(repr, context.metadata_field_info)) or "none"
    raise ValueError(
        f"metadata_fields_of names {store!r}, which is no vector store "
        f"(expected one of: {known})"
    )


def _refuse_undeclared_conditions(
    settings: GenerateSettings, schema: type[BaseModel], context: StageContext
) -> None:
    # A condition the LLM writes is held to a store's declared fields, as a
    # client's is (spec 7): a schema of conditions needs metadata_fields_of.
    held = condition_fields(schema)
    if not held or settings.metadata_fields_of is not None:
        return
    stores = ", ".join(map(repr, context.metadata_field_info)) or "none"
    raise ValueError(
        f"the output schema {settings.output_schema!r} holds conditions in {held}, "
        "which no vector store's declared fields would limit (expected "
        f"metadata_fields_of: <a vector store> (one of: {stores}) on the stage)"
    )


def _extra_variables(field_info: Sequence[MetadataFieldInfo] | None) -> set[str]:
    # The variables a prompt gets besides the fields its stage reads.
    return set() if field_info is None else {METADATA_FIELD_INFO_VARIABLE}


def _asked_schema(
    schema: type[BaseModel] | None, field_info: Sequence[MetadataFieldInfo] | None
) -> type[BaseModel] | None:
    # The schema the LLM is asked for: narrowed to the store's fields, if named.
    if schema is None or field_info is None:
        return schema
    return narrowed_to(schema, field_info)


def _named_schema(name: str) -> type[BaseModel]:
    # The schema a stage names, refused with the registered ones.
    if name in OUTPUT_SCHEMAS.names():
        return OUTPUT_SCHEMAS.get(name)
    known = ", ".join(map(repr, OUTPUT_SCHEMAS.names())) or "none"
    raise ValueError(
        f"names the output schema {name!r}, which is not registered "
        f"(expected one of: {known})"
    )


def _instance_of(schema: type[BaseModel], text: str) -> BaseModel:
    # The LLM's port promises an instance; an implementation that breaks the
    # promise fails the request as the LLM, showing what it answered.
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or '<answer>'}: {error['msg']}"
            for error in exc.errors()
        )
        raise RagDependencyError(
            "llm",
            f"the answer is not an instance of the output schema {schema.__name__} "
            f"({problems}; expected one JSON object it accepts): {text!r}",
        ) from exc
