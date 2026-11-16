"""The start-up checks of conditional edges: what they read, and the rows of their
``path_map`` (ADR 0007), on the fixture graphs of ``graph_check_fixtures``.

A classify stage writes ``kind``, a label: a question or a statement.
"""

from __future__ import annotations

from typing import Any

import pytest

from graph_check_fixtures import (
    CLASSIFY,
    DRAFT,
    POLISH,
    asking,
    check,
    graph_of,
)
from query_processing.application.edge_check import field_labels
from query_processing.application.graph_check import WorkflowGraphError
from query_processing.application.stages.base_stage import FieldTypes


def graph_with_label_edge(**edge: Any) -> Any:
    """START -> classify -?-> draft or polish; the label edge reads ``kind``."""
    label_edge = {
        "source": "classify",
        "path": "label",
        "reads": ["kind"],
        "path_map": {"question": "draft", "statement": "END"},
        **edge,
    }
    return graph_of(
        {"classify": CLASSIFY, "draft": DRAFT, "polish": POLISH},
        [["START", "classify"], ["draft", "polish", "END"]],
        [label_edge],
    )


def test_a_label_edge_with_a_row_for_each_label_passes() -> None:
    check(graph_with_label_edge())  # kind is read by the label edge alone


def test_a_path_map_must_list_exactly_the_labels_of_the_field() -> None:
    graph = graph_with_label_edge(path_map={"question": "draft"})  # no "statement"

    with pytest.raises(WorkflowGraphError) as refused:
        check(graph)

    assert str(refused.value) == (
        "conditional edge from 'classify' on 'label': path_map lists 'question' "
        "(expected one row for each label 'kind' takes, and no other: "
        "'question', 'statement')"
    )


def test_a_label_edge_with_a_true_row_is_refused_since_true_is_a_yes() -> None:
    graph = graph_with_label_edge(path_map={"question": "draft", "true": "END"})

    with pytest.raises(WorkflowGraphError, match="has a true or a false row"):
        check(graph)


def test_a_label_edge_on_a_field_that_declares_no_labels_is_refused() -> None:
    graph = graph_with_label_edge(reads=["conversational_query"])

    with pytest.raises(WorkflowGraphError, match="which has no declared labels"):
        check(graph)


def test_a_yes_or_no_edge_with_a_row_for_a_label_is_refused() -> None:
    edge = asking(
        "draft", "is_question", {True: "polish", False: "END", "maybe": "END"}
    )
    graph = graph_of({"draft": DRAFT, "polish": POLISH}, [["START", "draft"]], [edge])

    with pytest.raises(WorkflowGraphError, match="has rows for 'maybe', which"):
        check(graph)


def test_an_edge_must_read_one_field_for_each_value_its_condition_takes() -> None:
    edge = {**asking("draft", "is_question", {True: "polish", False: "END"})}
    edge["reads"] = ["conversational_query", "draft"]
    graph = graph_of({"draft": DRAFT, "polish": POLISH}, [["START", "draft"]], [edge])

    with pytest.raises(WorkflowGraphError) as refused:
        check(graph)

    assert str(refused.value) == (
        "conditional edge from 'draft' on 'is_question' reads 2 fields "
        "['conversational_query', 'draft'], but 'is_question' takes "
        "(conversational_query) (expected reads to list 1 field, one for each)"
    )


def test_an_edge_reads_only_what_is_set_once_its_source_has_run() -> None:
    reads_draft = {**asking("START", "is_long", {True: "draft", False: "END"})}
    reads_draft["reads"] = ["draft"]  # draft is written after START
    after_draft = {
        **reads_draft,
        "source": "draft",
        "path_map": {True: "polish", False: "END"},
    }
    stages = {"draft": DRAFT, "polish": POLISH}

    with pytest.raises(WorkflowGraphError, match="conditional edge from 'START' on"):
        check(graph_of(stages, [["polish", "END"]], [reads_draft]))
    check(graph_of(stages, [["START", "draft"], ["polish", "END"]], [after_draft]))


def test_an_edge_condition_is_still_not_used_twice_on_one_source() -> None:
    edge = asking("START", "is_long", {True: "draft", False: "END"})
    graph = graph_of(
        {"draft": DRAFT, "polish": POLISH}, [["draft", "polish"]], [edge, edge]
    )

    with pytest.raises(WorkflowGraphError, match="'is_long' is used twice on 'START'"):
        check(graph)


def test_two_writers_that_declare_other_labels_for_one_field_are_refused() -> None:
    fields = {
        "a": FieldTypes({}, {"kind": str}, {"kind": ("question",)}),
        "b": FieldTypes({}, {"kind": str}, {"kind": ("statement",)}),
    }

    with pytest.raises(WorkflowGraphError, match="the same labels"):
        field_labels(fields)
