"""The start-up checks of conditional edges: what they read, and the rows of their
``path_map`` (ADR 0007), on the fixture graphs of ``graph_check_fixtures``.

A classify stage writes ``kind``, a label: a question or a statement.
"""

from __future__ import annotations

from typing import Any

import pytest

from graph_check_fixtures import (
    CLASSIFY,
    COMPONENTS,
    DRAFT,
    FIXTURE_COMPONENT_TYPES,
    FIXTURE_STAGE_TYPES,
    JUDGE,
    POLISH,
    asking,
    check,
    graph_of,
)
from query_processing.application.edge_check import field_labels
from query_processing.application.edge_conditions.edge_condition import EdgeCondition
from query_processing.application.graph_check import WorkflowGraphError, check_graph
from query_processing.application.stages.base_stage import FieldTypes, StageContext
from query_processing.domain.registry import Registry


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


def test_the_reason_an_edge_condition_is_used_once_is_every_orchestrator_s() -> None:
    edge = asking("START", "is_long", {True: "draft", False: "END"})
    graph = graph_of(
        {"draft": DRAFT, "polish": POLISH}, [["draft", "polish"]], [edge, edge]
    )

    with pytest.raises(WorkflowGraphError) as refused:
        check(graph)

    assert "LangGraph names" not in str(refused.value)
    assert (
        "(expected each edge condition once per source: every orchestrator names "
        "the branch, or the step that asks it, after its edge condition, and one "
        "conditional edge already says where each of its answers leads)"
    ) in str(refused.value)


def test_two_writers_that_declare_other_labels_for_one_field_are_refused() -> None:
    fields = {
        "a": FieldTypes({}, {"kind": str}, {"kind": ("question",)}),
        "b": FieldTypes({}, {"kind": str}, {"kind": ("statement",)}),
    }

    with pytest.raises(WorkflowGraphError, match="the same labels"):
        field_labels(fields)


def test_a_writer_that_declares_no_labels_for_a_labelled_field_is_refused() -> None:
    fields = {
        "a": FieldTypes({}, {"kind": str}, {"kind": ("question", "statement")}),
        "b": FieldTypes({}, {"kind": str}),
    }

    with pytest.raises(WorkflowGraphError) as refused:
        field_labels(fields)

    assert str(refused.value) == (
        "'kind' is written with the labels 'question', 'statement' by 'a' and "
        "with no labels by 'b' (expected every writer of a field to declare the "
        "same labels)"
    )


def test_a_labelled_field_the_request_provides_is_refused() -> None:
    fields = {"a": FieldTypes({}, {"mode": str}, {"mode": ("query",)})}

    with pytest.raises(WorkflowGraphError, match="with no labels by the request"):
        field_labels(fields)


def unanswering(conversational_query):  # type: ignore[no-untyped-def]
    return conversational_query


def test_an_edge_condition_that_states_no_answer_is_refused_at_start_up() -> None:
    conditions: Registry[EdgeCondition] = Registry("edge condition")
    conditions.register("unanswering")(unanswering)
    edge = asking("draft", "unanswering", {True: "polish", False: "END"})
    graph = graph_of({"draft": DRAFT, "polish": POLISH}, [["START", "draft"]], [edge])

    with pytest.raises(WorkflowGraphError) as refused:
        check_graph(
            graph,
            COMPONENTS,
            StageContext(),
            component_types=FIXTURE_COMPONENT_TYPES,
            stage_types=FIXTURE_STAGE_TYPES,
            edge_conditions=conditions,
        )

    assert str(refused.value).startswith(
        "conditional edge from 'draft' on 'unanswering': the edge condition "
        "'unanswering' states no return type (expected -> bool"
    )


def test_an_edge_reads_a_field_only_as_the_type_its_condition_takes() -> None:
    edge = {**asking("draft", "is_question", {True: "polish", False: "END"})}
    edge["reads"] = ["verdict"]  # a bool, where is_question takes a str
    stages = {"draft": DRAFT, "judge": JUDGE, "polish": POLISH}

    with pytest.raises(WorkflowGraphError) as refused:
        check(
            graph_of(
                stages, [["START", "draft", "judge"]], [edge | {"source": "judge"}]
            )
        )

    assert str(refused.value) == (
        "conditional edge from 'judge' on 'is_question' reads 'verdict' as str, "
        "but 'judge' writes it as bool (expected a field written as str)"
    )
