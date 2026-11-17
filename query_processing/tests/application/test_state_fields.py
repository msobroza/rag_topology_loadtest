"""The workflow state is built from the fields the request and the stages write."""

from __future__ import annotations

import collections.abc
from typing import Any, Literal, Optional, get_type_hints

import pytest
from langgraph.graph import END, START, StateGraph

from query_processing.application.stages.base_stage import FieldTypes
from query_processing.application.state_fields import (
    WorkflowGraphError,
    add_conditions,
    annotation_accepts,
    field_types_of,
    state_type,
    type_fits,
)
from query_processing.application.workflow_state import (
    METADATA_FILTER_FIELD,
    REQUEST_FIELD_TYPES,
    initial_state,
    merge_by_stage,
)
from query_processing.domain.conversation import HistoryMessage
from query_processing.domain.metadata_filter import MetadataCondition


def test_the_state_type_holds_the_fields_the_timings_and_the_tokens() -> None:
    hints = get_type_hints(state_type({"draft": str}), include_extras=True)

    assert hints["draft"] is str
    assert hints["timings_ms"].__metadata__ == (merge_by_stage,)
    assert hints["tokens"].__metadata__ == (merge_by_stage,)


def test_the_fields_are_the_requests_and_those_the_stages_write() -> None:
    stages = {
        "draft": FieldTypes({"conversational_query": str}, {"draft": str}),
        "judge": FieldTypes({"draft": str}, {"verdict": bool}),
    }

    types = field_types_of(stages)

    assert types == {**REQUEST_FIELD_TYPES, "draft": str, "verdict": bool}


def test_a_writer_of_any_type_takes_the_type_of_the_other_writer() -> None:
    stages = {
        "loose": FieldTypes({}, {"draft": Any}),
        "draft": FieldTypes({}, {"draft": str}),
    }

    assert field_types_of(stages)["draft"] is str


def test_two_writers_that_disagree_are_named() -> None:
    stages = {
        "draft": FieldTypes({}, {"draft": str}),
        "judge": FieldTypes({}, {"draft": bool}),
    }

    with pytest.raises(WorkflowGraphError) as refused:
        field_types_of(stages)

    assert str(refused.value) == (
        "'draft' is written as str by 'draft' and as bool by 'judge' (expected "
        "every writer of a field to write one type)"
    )


def test_a_stage_that_rewrites_a_request_field_must_keep_its_type() -> None:
    stages = {"rewrite": FieldTypes({}, {"standalone_query": list[str]})}

    with pytest.raises(WorkflowGraphError, match="as str by the request and as"):
        field_types_of(stages)


@pytest.mark.parametrize(
    ("given", "takes"),
    [
        (list[float], list[float]),
        (list[float], list[float] | None),  # an attribute that accepts None
        (str, Optional[str]),  # noqa: UP045 - the older spelling of the same
        (str, Any),
        (Any, int),
    ],
)
def test_an_attribute_takes_a_field_of_its_type_or_of_a_member_of_its_union(
    given: Any, takes: Any
) -> None:
    assert type_fits(given, takes)
    assert annotation_accepts(given, takes)


@pytest.mark.parametrize(
    ("given", "takes"),
    [
        (str, int),
        (list[float], list[str]),
        (str | None, str),  # None would reach a reader that cannot take it
        (list[HistoryMessage], str),
        (Literal["query"], int),
        (list[float], str | None),
    ],
)
def test_a_field_fits_no_reader_of_another_type(given: Any, takes: Any) -> None:
    assert not type_fits(given, takes)
    assert not annotation_accepts(given, takes)


@pytest.mark.parametrize(
    ("given", "annotation"),
    [
        (list[HistoryMessage], collections.abc.Sequence[object]),
        (Literal["query", "retrieve"], str),
        (bool, object),
        (Literal["query"], str | None),
    ],
)
def test_an_annotation_also_takes_a_field_of_a_narrower_type(
    given: Any, annotation: Any
) -> None:
    assert annotation_accepts(given, annotation)
    assert not type_fits(given, annotation)  # an attribute takes its own type


ON_PAY = MetadataCondition(attribute="topic", comparator="eq", value="pay")
ON_LEAVE = MetadataCondition(attribute="topic", comparator="ne", value="leave")


def test_add_conditions_adds_and_never_removes() -> None:
    assert add_conditions([ON_PAY], [ON_LEAVE]) == [ON_PAY, ON_LEAVE]
    assert add_conditions([ON_PAY], [ON_PAY, ON_LEAVE]) == [ON_PAY, ON_LEAVE]
    assert add_conditions([ON_PAY], []) == [ON_PAY]


async def test_a_stage_that_writes_the_metadata_filter_adds_conditions() -> None:
    graph = StateGraph(state_type(field_types_of({})))
    graph.add_node("narrow", lambda state: {METADATA_FILTER_FIELD: [ON_LEAVE]})
    graph.add_edge(START, "narrow")
    graph.add_edge("narrow", END)

    final = await graph.compile().ainvoke(
        initial_state("q?", "retrieve", metadata_filter=[ON_PAY])
    )

    assert final[METADATA_FILTER_FIELD] == [ON_PAY, ON_LEAVE]
