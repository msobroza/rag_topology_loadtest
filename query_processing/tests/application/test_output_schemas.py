"""A ``generate`` stage that names an output schema writes each of its fields."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from output_schema_fakes import FIXTURE_SCHEMA, FixtureRewrite
from query_processing.adapters.llm.base_llm import BaseLLM
from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.application.output_schemas import OUTPUT_SCHEMAS
from query_processing.application.stages.base_stage import StageContext, StageFields
from query_processing.application.stages.generate_stage import (
    GenerateSettings,
    GenerateStage,
)
from query_processing.domain.errors import RagDependencyError
from query_processing.domain.metadata_filter import MetadataCondition
from query_processing.domain.prompts import PromptMessage

REWRITE = GenerateSettings(prompt="rewrite", output_schema=FIXTURE_SCHEMA)
CONTEXT = StageContext(
    {"rewrite": [PromptMessage(role="user", content="{{ conversational_query }}")]}
)
# The schema's fields, stated in another order than the schema's.
FIELDS = StageFields(("conversational_query",), ("metadata_filter", "standalone_query"))
ON_PAY = {"attribute": "topic", "comparator": "eq", "value": "pay"}


class ScriptedLLM(BaseLLM):
    """Answers one text, and records the schema each call asked for."""

    model_name = "scripted-llm"

    def __init__(self, answer: str) -> None:
        self.answer, self.schemas = answer, list[type[BaseModel] | None]()

    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
    ) -> str:
        self.schemas.append(output_schema)
        return self.answer


def test_a_registered_schema_is_found_by_its_name() -> None:
    assert OUTPUT_SCHEMAS.get(FIXTURE_SCHEMA) is FixtureRewrite


def test_a_stage_with_a_schema_writes_the_schemas_fields_with_their_types() -> None:
    field_types = GenerateStage.check_fields(REWRITE, FIELDS, CONTEXT)

    assert field_types.reads == {"conversational_query": Any}
    assert field_types.writes == {
        "metadata_filter": list[MetadataCondition],
        "standalone_query": str,
    }


@pytest.mark.parametrize(
    ("settings", "writes", "refused"),
    [
        (
            GenerateSettings(prompt="rewrite", output_schema="fixture_rewrit"),
            FIELDS.writes,
            r"names the output schema 'fixture_rewrit', which is not registered "
            r"\(expected one of: .*'fixture_rewrite'",
        ),
        (
            REWRITE,
            ("standalone_query",),
            r"the output schema 'fixture_rewrite' has the fields \['standalone_query', "
            r"'metadata_filter'\], but the stage writes \['standalone_query'\] "
            r"\(expected writes listing each field of the schema, by name\)",
        ),
        (
            REWRITE,
            ("standalone_query", "filter"),
            r"but the stage writes \['standalone_query', 'filter'\]",
        ),
    ],
    ids=["unknown-schema", "a-field-missing", "a-field-misnamed"],
)
def test_a_stage_with_a_schema_refuses_what_does_not_fit_it(
    settings: GenerateSettings, writes: tuple[str, ...], refused: str
) -> None:
    fields = StageFields(FIELDS.reads, writes)

    with pytest.raises(ValueError, match=refused):
        GenerateStage.check_fields(settings, fields, CONTEXT)


async def test_a_stage_with_a_schema_writes_each_field_of_the_answer() -> None:
    llm = ScriptedLLM(
        f'{{"standalone_query": "pay in May", "metadata_filter": [{_json(ON_PAY)}]}}'
    )
    stage = GenerateStage(llm, REWRITE, FIELDS, CONTEXT)

    update = await stage.run({"conversational_query": "and in May?"})

    assert update == {
        "standalone_query": "pay in May",
        "metadata_filter": [MetadataCondition.model_validate(ON_PAY)],
    }
    assert llm.schemas == [FixtureRewrite]


async def test_on_the_fake_llm_a_stage_with_a_schema_writes_its_fields() -> None:
    stage = GenerateStage(FakeLLM(), REWRITE, FIELDS, CONTEXT)

    update = await stage.run({"conversational_query": "and b?"})

    assert update == {"standalone_query": "[fake-llm] and b?", "metadata_filter": []}


async def test_an_answer_the_schema_refuses_fails_the_request_showing_it() -> None:
    stage = GenerateStage(ScriptedLLM("Sure! pay in May"), REWRITE, FIELDS, CONTEXT)

    with pytest.raises(RagDependencyError, match="Sure! pay in May") as info:
        await stage.run({"conversational_query": "and in May?"})
    assert info.value.component == "llm"
    assert "FixtureRewrite" in info.value.detail


async def test_without_a_schema_the_stage_writes_the_text_and_asks_for_none() -> None:
    llm = ScriptedLLM("pay in May")
    settings = GenerateSettings(prompt="rewrite")
    stage = GenerateStage(
        llm, settings, StageFields(FIELDS.reads, ("answer",)), CONTEXT
    )

    assert await stage.run({"conversational_query": "q"}) == {"answer": "pay in May"}
    assert llm.schemas == [None]


def _json(condition: dict[str, str]) -> str:
    return MetadataCondition.model_validate(condition).model_dump_json()
