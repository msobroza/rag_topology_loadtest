"""A ``generate`` stage with ``metadata_fields_of``: the conditions it may write.

The stage names a vector store; the schema the LLM is asked for allows only
conditions on that store's declared fields, and the prompt is given them.
"""

from __future__ import annotations

import json

import pytest
from query_processing_core.retrieval.metadata_filter import (
    MetadataCondition,
    MetadataFieldInfo,
)

from output_schema_fakes import ScriptedLLM
from query_processing.application.output_schemas import StandaloneQueryWithFilter
from query_processing.application.stages.base_stage import StageContext, StageFields
from query_processing.application.stages.generate_stage import (
    GenerateSettings,
    GenerateStage,
)
from query_processing.domain.errors import RagDependencyError
from query_processing.domain.prompts import PromptMessage

SOURCE = MetadataFieldInfo(
    name="source",
    type="string",
    description="who published it",
    allowed_values=["AMF", "ACPR"],
)
REWRITE = GenerateSettings(
    prompt="rewrite",
    output_schema="standalone_query_with_filter",
    metadata_fields_of="handbook",
)
LISTED = (
    "{% for field in metadata_field_info %}{{ field.name }}: "
    "{{ field.allowed_values | join(', ') }}{% endfor %}|{{ conversational_query }}"
)
CONTEXT = StageContext(
    {"rewrite": [PromptMessage(role="user", content=LISTED)]},
    metadata_field_info={"handbook": [SOURCE], "archive": []},
)
FIELDS = StageFields(("conversational_query",), ("standalone_query", "metadata_filter"))
ON_AMF = {"attribute": "source", "comparator": "in", "value": ["AMF"]}


def _answer(*conditions: dict[str, object]) -> str:
    return json.dumps({"standalone_query": "AMF on it", "metadata_filter": conditions})


async def test_the_prompt_is_given_the_fields_and_the_llm_the_narrowed_schema() -> None:
    llm = ScriptedLLM(_answer(ON_AMF))
    stage = GenerateStage(llm, REWRITE, FIELDS, CONTEXT)

    update = await stage.run({"conversational_query": "and the AMF?"})

    # The state holds the domain's conditions, not the narrowed schema's models.
    assert update == {
        "standalone_query": "AMF on it",
        "metadata_filter": [MetadataCondition.model_validate(ON_AMF)],
    }
    assert llm.messages[0][0]["content"] == "source: AMF, ACPR|and the AMF?"
    (asked,) = llm.schemas
    assert asked is not None
    assert asked is not StandaloneQueryWithFilter
    assert "ACPR" in json.dumps(asked.model_json_schema())


async def test_a_condition_on_an_undeclared_value_fails_the_request() -> None:
    saaq = {"attribute": "source", "comparator": "eq", "value": "SAAQ"}
    stage = GenerateStage(ScriptedLLM(_answer(saaq)), REWRITE, FIELDS, CONTEXT)

    with pytest.raises(RagDependencyError, match="SAAQ"):
        await stage.run({"conversational_query": "and the SAAQ?"})


@pytest.mark.parametrize(
    ("store", "refused"),
    [
        (
            "handbok",
            r"metadata_fields_of names 'handbok', which is no vector store "
            r"\(expected one of: 'handbook', 'archive'\)",
        ),
        ("archive", "at least one declared field"),
    ],
    ids=["no-such-store", "a-store-that-declares-nothing"],
)
def test_start_up_refuses_a_store_that_cannot_narrow_the_schema(
    store: str, refused: str
) -> None:
    settings = REWRITE.model_copy(update={"metadata_fields_of": store})

    with pytest.raises(ValueError, match=refused):
        GenerateStage.check_fields(settings, FIELDS, CONTEXT)


def test_start_up_refuses_a_schema_of_conditions_without_metadata_fields_of() -> None:
    # Regression of the review: the stage wrote any condition the LLM answered,
    # on any field, where a client's would get a 422 (spec 7).
    settings = REWRITE.model_copy(update={"metadata_fields_of": None})
    plain = [PromptMessage(role="user", content="{{ conversational_query }}")]
    context = StageContext({"rewrite": plain}, CONTEXT.metadata_field_info)

    with pytest.raises(
        ValueError,
        match=r"the output schema 'standalone_query_with_filter' holds conditions "
        r"in \['metadata_filter'\].*expected metadata_fields_of: <a vector store> "
        r"\(one of: 'handbook', 'archive'\)",
    ):
        GenerateStage.check_fields(settings, FIELDS, context)


def test_without_metadata_fields_of_the_prompt_cannot_use_the_fields() -> None:
    settings = REWRITE.model_copy(update={"metadata_fields_of": None})

    with pytest.raises(ValueError, match="uses 'metadata_field_info'"):
        GenerateStage.check_fields(settings, FIELDS, CONTEXT)
