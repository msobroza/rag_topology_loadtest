"""A metadata filter's conditions, checked against the fields a vector store declares."""

from __future__ import annotations

from typing import Any

import pytest

from query_processing_core.retrieval.metadata_filter import (
    MetadataCondition,
    MetadataFieldInfo,
    check_condition_shapes,
    check_conditions,
)

SOURCE = MetadataFieldInfo(
    name="source", type="string", description="d", allowed_values=["AMF", "GAA"]
)
ACTIVE = MetadataFieldInfo(name="active", type="boolean", description="d")
SEQUENCE = MetadataFieldInfo(name="sequence", type="integer", description="d")
FIELDS = [SOURCE, ACTIVE, SEQUENCE]


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
            {"attribute": "sequence", "comparator": "eq", "value": [1, 2]},
            "'eq' takes one value",
        ),
        (
            {"attribute": "source", "comparator": "in", "value": []},
            "'in' takes a list of at least one value",
        ),
        (
            {"attribute": "sequence", "comparator": "nin", "value": []},
            "'nin' takes a list of at least one value",
        ),
        (
            {"attribute": "sequence", "comparator": "gte", "value": "two"},
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
    assert "'source', 'active', 'sequence'" in str(refused.value)


@pytest.mark.parametrize(
    "condition",
    [
        {"attribute": "source", "comparator": "in", "value": ["AMF", "GAA"]},
        {"attribute": "source", "comparator": "ne", "value": "GAA"},
        {"attribute": "active", "comparator": "eq", "value": True},
        {"attribute": "sequence", "comparator": "lt", "value": 3},
        {"attribute": "sequence", "comparator": "nin", "value": [1, 2]},
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
    assert MetadataCondition(
        attribute="sequence", comparator="in", value=[2]
    ).value == [2]


@pytest.mark.parametrize(
    ("condition", "problem"),
    [
        (
            {"attribute": "active", "comparator": "in", "value": True},
            "'in' takes a list",
        ),
        (
            {"attribute": "version", "comparator": "eq", "value": ["1"]},
            "'eq' takes one",
        ),
        ({"attribute": "version", "comparator": "nin", "value": []}, "at least one"),
    ],
)
def test_a_condition_of_the_wrong_shape_is_refused_without_a_field_declared(
    condition: dict[str, Any], problem: str
) -> None:
    # A prefilter names fields no store declares: only its shape is checked.
    with pytest.raises(ValueError, match=problem) as refused:
        check_condition_shapes([MetadataCondition(**condition)], "prefilter")

    assert str(refused.value).startswith("prefilter[0] ")


def test_a_condition_of_the_right_shape_needs_no_field_declared() -> None:
    check_condition_shapes(
        [
            MetadataCondition(attribute="version", comparator="eq", value="1"),
            # A boolean has eq and ne only, in the domain's table: no "in".
            MetadataCondition(attribute="version", comparator="in", value=["1", "2"]),
            MetadataCondition(attribute="active", comparator="ne", value=False),
            MetadataCondition(attribute="year", comparator="gte", value=2020),
            MetadataCondition(attribute="score", comparator="lt", value=0.5),
        ],
        "prefilter",
    )


@pytest.mark.parametrize(
    ("condition", "problem"),
    [
        (
            {"attribute": "version", "comparator": "gt", "value": "1"},
            r"'gt' is not a comparator of a string \(expected one of: eq, in, ne, nin\)",
        ),
        (
            {"attribute": "active", "comparator": "gte", "value": True},
            r"'gte' is not a comparator of a boolean \(expected one of: eq, ne\)",
        ),
        (
            {"attribute": "version", "comparator": "in", "value": ["1", 2, True]},
            r"\['1', 2, True\] mixes values of the types string, integer, boolean "
            r"\(expected values of one type\)",
        ),
        (
            # Chroma refuses it too: its values must all have one type.
            {"attribute": "score", "comparator": "nin", "value": [1, 2.5]},
            r"mixes values of the types integer, float",
        ),
    ],
    ids=[
        "gt-on-a-string",
        "gte-on-a-boolean",
        "in-of-mixed-types",
        "nin-of-an-integer-and-a-float",
    ],
)
def test_a_prefilter_condition_is_refused_when_its_comparator_misfits_its_value(
    condition: dict[str, Any], problem: str
) -> None:
    # Regression of the review: these were accepted, the deploy tool's check
    # included, and every Chroma search then failed.
    with pytest.raises(ValueError, match=problem) as refused:
        check_condition_shapes([MetadataCondition(**condition)], "prefilter")

    assert str(refused.value).startswith("prefilter[0] ")
