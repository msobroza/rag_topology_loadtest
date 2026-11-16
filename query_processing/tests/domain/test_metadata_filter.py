"""A metadata filter's conditions, checked against the fields a vector store declares."""

from __future__ import annotations

from typing import Any

import pytest

from query_processing.domain.metadata_filter import (
    MetadataCondition,
    MetadataFieldInfo,
    check_conditions,
)

SOURCE = MetadataFieldInfo(
    name="source", type="string", description="d", allowed_values=["AMF", "GAA"]
)
ACTIVE = MetadataFieldInfo(name="active", type="boolean", description="d")
CHUNK = MetadataFieldInfo(name="chunk", type="integer", description="d")
FIELDS = [SOURCE, ACTIVE, CHUNK]


@pytest.mark.parametrize(
    ("condition", "problem"),
    [
        (
            {"attribute": "sorce", "comparator": "eq", "value": "AMF"},
            "no field 'sorce'",
        ),
        (
            {"attribute": "source", "comparator": "eq", "value": "SAAQ"},
            "'SAAQ' is not one of",
        ),
        (
            {"attribute": "source", "comparator": "nin", "value": ["AMF", "SAAQ"]},
            "'SAAQ' is not one of",
        ),
        (
            {"attribute": "source", "comparator": "gt", "value": "AMF"},
            "'gt' is not a comparator of a string",
        ),
        (
            {"attribute": "active", "comparator": "lte", "value": True},
            "'lte' is not a comparator of a boolean",
        ),
        (
            {"attribute": "source", "comparator": "in", "value": "AMF"},
            "'in' takes a list",
        ),
        (
            {"attribute": "chunk", "comparator": "eq", "value": [1, 2]},
            "'eq' takes one value",
        ),
        (
            {"attribute": "chunk", "comparator": "gte", "value": "two"},
            "'two' is not of type integer",
        ),
        (
            {"attribute": "active", "comparator": "eq", "value": 1},
            "1 is not of type boolean",
        ),
    ],
)
def test_a_condition_that_does_not_fit_its_field_is_refused(
    condition: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        check_conditions([MetadataCondition(**condition)], FIELDS)


def test_the_refusal_names_the_condition_and_the_fields_declared() -> None:
    with pytest.raises(ValueError, match="sorce") as refused:
        check_conditions(
            [MetadataCondition(attribute="sorce", comparator="eq", value="AMF")],
            FIELDS,
        )

    assert "sorce" in str(refused.value)
    assert "'source', 'active', 'chunk'" in str(refused.value)


@pytest.mark.parametrize(
    "condition",
    [
        {"attribute": "source", "comparator": "in", "value": ["AMF", "GAA"]},
        {"attribute": "source", "comparator": "ne", "value": "GAA"},
        {"attribute": "active", "comparator": "eq", "value": True},
        {"attribute": "chunk", "comparator": "lt", "value": 3},
        {"attribute": "chunk", "comparator": "nin", "value": [1, 2]},
    ],
)
def test_a_condition_that_fits_its_field_is_accepted(condition: dict[str, Any]) -> None:
    check_conditions([MetadataCondition(**condition)], FIELDS)


def test_a_number_field_takes_an_integer_or_a_float() -> None:
    score = MetadataFieldInfo(name="score", type="float", description="d")

    check_conditions(
        [MetadataCondition(attribute="score", comparator="gt", value=1)], [score]
    )
    with pytest.raises(ValueError, match="True is not of type float"):
        check_conditions(
            [MetadataCondition(attribute="score", comparator="gt", value=True)], [score]
        )


def test_an_empty_filter_needs_no_declared_field() -> None:
    check_conditions([], [])


def test_a_condition_keeps_the_type_of_its_value() -> None:
    condition = MetadataCondition.model_validate(
        {"attribute": "active", "comparator": "eq", "value": True}
    )

    assert condition.value is True
    assert MetadataCondition(attribute="chunk", comparator="in", value=[2]).value == [2]
