"""A ``generate`` stage that names an output schema writes each of its fields."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from output_schema_fakes import FIXTURE_SCHEMA, FixtureRewrite, ScriptedLLM
from query_processing.adapters.llm.base_llm import BaseLLM
from query_processing.adapters.llm.fake.fake_llm import FakeLLM, fake_instance_json
from query_processing.application.output_schemas import (
    OUTPUT_SCHEMAS,
    StandaloneQueryWithFilter,
    narrowed_to,
)
from query_processing.application.stages.base_stage import StageContext, StageFields
from query_processing.application.stages.generate_stage import (
    GenerateSettings,
    GenerateStage,
)
from query_processing.domain.errors import RagDependencyError
from query_processing.domain.metadata_filter import (
    MetadataCondition,
    MetadataFieldInfo,
)
from query_processing.domain.prompts import PromptMessage

# A schema of conditions names the store whose declared fields limit them.
REWRITE = GenerateSettings(
    prompt="rewrite", output_schema=FIXTURE_SCHEMA, metadata_fields_of="handbook"
)
TOPIC = MetadataFieldInfo(name="topic", type="string", description="what it is about")
CONTEXT = StageContext(
    {"rewrite": [PromptMessage(role="user", content="{{ conversational_query }}")]},
    metadata_field_info={"handbook": [TOPIC]},
)
# The schema's fields, stated in another order than the schema's.
FIELDS = StageFields(("conversational_query",), ("metadata_filter", "standalone_query"))
ON_PAY = {"attribute": "topic", "comparator": "eq", "value": "pay"}
SOURCE = MetadataFieldInfo(
    name="source",
    type="string",
    description="who published the document",
    allowed_values=["AMF", "ACPR"],
)
YEAR = MetadataFieldInfo(
    name="year", type="integer", description="when it was published"
)
_DESCRIBED, _ALIASED = "fixture_described", "fixture_aliased"


@OUTPUT_SCHEMAS.register(_DESCRIBED)
class _Described(BaseModel):
    # A use case's schema: a field with a constraint, a description, an alias.
    model_config = ConfigDict(extra="forbid")

    standalone_query: str = Field(
        min_length=5, description="The message, self-contained.", alias="query"
    )
    metadata_filter: list[MetadataCondition]


class _Aliased(BaseModel):
    # A schema of no conditions, asked for as it is, with an alias.
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(alias="finalAnswer")


OUTPUT_SCHEMAS.register(_ALIASED)(_Aliased)


def test_a_registered_schema_is_found_by_its_name() -> None:
    assert OUTPUT_SCHEMAS.get(FIXTURE_SCHEMA) is FixtureRewrite


def test_the_toolkit_registers_a_standalone_query_with_a_metadata_filter() -> None:
    schema = OUTPUT_SCHEMAS.get("standalone_query_with_filter")

    assert schema is StandaloneQueryWithFilter
    assert {name: info.annotation for name, info in schema.model_fields.items()} == {
        "standalone_query": str,
        "metadata_filter": list[MetadataCondition],
    }


def test_the_schema_is_narrowed_to_the_declared_fields_and_values() -> None:
    narrowed = narrowed_to(StandaloneQueryWithFilter, [SOURCE])
    kept = {
        "standalone_query": "q",
        "metadata_filter": [
            {"attribute": "source", "comparator": "in", "value": ["AMF"]}
        ],
    }

    assert narrowed.model_validate(kept).metadata_filter[0].value == ["AMF"]
    # A field that is no list of conditions keeps its own type, inline.
    assert '"$ref"' not in json.dumps(
        narrowed.model_json_schema()["properties"]["standalone_query"]
    )


@pytest.mark.parametrize(
    "condition",
    [
        {"attribute": "sorce", "comparator": "eq", "value": "AMF"},
        {"attribute": "source", "comparator": "eq", "value": "SAAQ"},
        {"attribute": "source", "comparator": "gt", "value": "AMF"},
        {"attribute": "source", "comparator": "in", "value": "AMF"},
        # Regression of the review: an empty list reached the store, which may
        # refuse it mid-search, as check_conditions says.
        {"attribute": "source", "comparator": "in", "value": []},
        {"attribute": "source", "comparator": "eq", "value": "AMF", "why": "x"},
    ],
    ids=[
        "no-such-field",
        "a-value-not-allowed",
        "a-comparator-a-string-lacks",
        "one-value-for-in",
        "an-empty-list-for-in",
        "an-extra-key",
    ],
)
def test_the_narrowed_schema_refuses_a_condition_the_store_does_not_allow(
    condition: dict[str, object],
) -> None:
    narrowed = narrowed_to(StandaloneQueryWithFilter, [SOURCE])

    with pytest.raises(ValidationError):
        narrowed.model_validate(
            {"standalone_query": "q", "metadata_filter": [condition]}
        )


def test_a_number_field_keeps_the_order_comparators_and_takes_numbers() -> None:
    narrowed = narrowed_to(StandaloneQueryWithFilter, [SOURCE, YEAR])
    after = {"attribute": "year", "comparator": "gte", "value": 2020}

    answer = narrowed.model_validate(
        {"standalone_query": "q", "metadata_filter": [after]}
    )

    assert answer.metadata_filter[0].value == 2020
    with pytest.raises(ValidationError):
        narrowed.model_validate(
            {"standalone_query": "q", "metadata_filter": [dict(after, value="2020")]}
        )


def test_a_float_field_takes_an_integer_as_check_conditions_does() -> None:
    price = MetadataFieldInfo(name="price", type="float", description="in euros")
    under = {"attribute": "price", "comparator": "lt", "value": 3}

    answer = narrowed_to(StandaloneQueryWithFilter, [price]).model_validate_json(
        json.dumps({"standalone_query": "q", "metadata_filter": [under]})
    )

    assert answer.metadata_filter[0].value == 3


def test_the_narrowed_schema_is_strict_no_extra_key_and_no_default() -> None:
    schema = narrowed_to(StandaloneQueryWithFilter, [SOURCE, YEAR]).model_json_schema()

    objects = [schema, *schema.get("$defs", {}).values()]
    assert all(o["additionalProperties"] is False for o in objects)
    assert all(set(o["required"]) == set(o["properties"]) for o in objects)
    assert not any("default" in p for o in objects for p in o["properties"].values())


def test_a_schema_cannot_be_narrowed_to_no_field() -> None:
    with pytest.raises(ValueError, match="at least one declared field"):
        narrowed_to(StandaloneQueryWithFilter, [])


def test_the_fake_llm_answers_an_instance_of_a_narrowed_schema() -> None:
    narrowed = narrowed_to(StandaloneQueryWithFilter, [SOURCE])

    answer = narrowed.model_validate_json(fake_instance_json(narrowed, "[fake-llm] q"))

    assert (answer.standalone_query, answer.metadata_filter) == ("[fake-llm] q", [])


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
    # The schema is asked for narrowed to the store's fields, under its name.
    assert [schema and schema.__name__ for schema in llm.schemas] == ["FixtureRewrite"]


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


def test_the_narrowed_schema_keeps_each_fields_constraints_and_description() -> None:
    # Regression of the review: the narrowing rebuilt each field without its
    # Field(...), so the constraint and the description left the schema sent.
    narrowed = narrowed_to(_Described, [SOURCE])

    sent = narrowed.model_json_schema()["properties"]["query"]
    assert sent["minLength"] == 5
    assert sent["description"] == "The message, self-contained."
    with pytest.raises(ValidationError):
        narrowed.model_validate({"query": "hi", "metadata_filter": []})


@pytest.mark.parametrize(
    ("llm", "schema", "writes", "written"),
    [
        (ScriptedLLM('{"finalAnswer": "a"}'), _ALIASED, ("answer",), ["a"]),
        (FakeLLM(), _ALIASED, ("answer",), ["[fake-llm] and in May?"]),
        (
            ScriptedLLM('{"query": "pay in May", "metadata_filter": []}'),
            _DESCRIBED,
            FIELDS.writes,
            [[], "pay in May"],
        ),
        (FakeLLM(), _DESCRIBED, FIELDS.writes, [[], "[fake-llm] and in May?"]),
    ],
    ids=["as-is", "as-is-on-the-fake-llm", "narrowed", "narrowed-on-the-fake-llm"],
)
async def test_a_schema_with_an_alias_writes_its_fields_by_their_names(
    llm: BaseLLM, schema: str, writes: tuple[str, ...], written: list[object]
) -> None:
    # Regression of the review: the answer was validated a second time from a
    # dump by field name, which a schema with an alias refuses.
    store = "handbook" if schema == _DESCRIBED else None
    settings = GenerateSettings(
        prompt="rewrite", output_schema=schema, metadata_fields_of=store
    )
    fields = StageFields(FIELDS.reads, writes)

    update = await GenerateStage(llm, settings, fields, CONTEXT).run(
        {"conversational_query": "and in May?"}
    )

    assert update == dict(zip(writes, written, strict=True))


async def test_a_narrowed_answer_that_breaks_a_constraint_fails_as_the_llm() -> None:
    llm = ScriptedLLM('{"query": "hi", "metadata_filter": []}')
    settings = GenerateSettings(
        prompt="rewrite", output_schema=_DESCRIBED, metadata_fields_of="handbook"
    )
    stage = GenerateStage(llm, settings, FIELDS, CONTEXT)

    with pytest.raises(RagDependencyError, match="at least 5 characters"):
        await stage.run({"conversational_query": "and in May?"})


def _json(condition: dict[str, str]) -> str:
    return MetadataCondition.model_validate(condition).model_dump_json()
