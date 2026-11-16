"""The workflow state is built from the fields the request and the stages write."""

from __future__ import annotations

from typing import Any, get_type_hints

import pytest

from query_processing.application.stages.base_stage import FieldTypes
from query_processing.application.state_fields import (
    WorkflowGraphError,
    field_types_of,
    state_type,
)
from query_processing.application.workflow_state import (
    REQUEST_FIELD_TYPES,
    merge_timings,
)


def test_the_state_type_holds_the_fields_and_the_timings() -> None:
    hints = get_type_hints(state_type({"draft": str}), include_extras=True)

    assert hints["draft"] is str
    assert hints["timings_ms"].__metadata__ == (merge_timings,)


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
