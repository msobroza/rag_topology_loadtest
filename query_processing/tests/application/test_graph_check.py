"""The start-up checks are generic: graphs of fixture stage types, fixture component
types and fixture edge conditions, in registries of the tests' own
(``graph_check_fixtures``).

The RAG's own graphs are checked through ``qp-serve`` in
``tests/entrypoints/start_up/test_wrong_graph_at_start_up.py``; the checks of the edges
are in ``test_edge_check.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from graph_check_fixtures import (
    ANOTHER_DRAFT,
    DRAFT,
    EXPLAIN,
    JUDGE,
    POLISH,
    WITH_VOICE,
    asking,
    check,
    graph_of,
)
from query_processing.application.graph_check import (
    WorkflowGraphError,
    fields_set_before,
)
from query_processing.application.rag_dependencies import RagDependencies
from query_processing.application.workflow_state import REQUEST_FIELDS
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.components.base_component import COMPONENT_TYPES


def test_a_graph_of_other_stage_types_and_edge_conditions_passes() -> None:
    graph = graph_of(
        {"draft": DRAFT, "polish": POLISH, "judge": JUDGE, "explain": EXPLAIN},
        [["START", "draft", "judge", "explain"], ["polish", "END"]],
        [
            asking("explain", "is_question", {True: "polish", False: "END"}),
            asking("draft", "is_question", {True: "END", False: "END"}),
        ],
    )

    check(graph)


def test_a_field_written_on_one_branch_only_is_refused() -> None:
    graph = graph_of(
        {"draft": DRAFT, "polish": POLISH},
        [["draft", "polish", "END"]],
        [asking("START", "is_question", {True: "draft", False: "polish"})],
    )

    with pytest.raises(WorkflowGraphError) as refused:
        check(graph)

    assert str(refused.value).startswith(
        "stage 'polish' reads the workflow state field 'draft', which the request "
        "does not provide and is written by 'draft', but not on every path from "
        "START to 'polish'"
    )


def test_a_field_written_on_every_branch_passes() -> None:
    graph = graph_of(
        {"short": ANOTHER_DRAFT, "long": ANOTHER_DRAFT, "polish": POLISH},
        [["short", "polish"], ["long", "polish", "END"]],
        [asking("START", "is_long", {True: "long", False: "short"})],
    )

    check(graph)


def test_a_stage_on_a_loop_reads_only_what_is_set_before_its_first_pass() -> None:
    # START -> polish -> draft -> polish ...: the first polish has no draft.
    looped = graph_of(
        {"draft": DRAFT, "polish": POLISH},
        [["START", "polish", "draft"]],
        [asking("draft", "is_long", {True: "polish", False: "END"})],
    )
    # START -> draft -> polish -> draft ...: every pass of polish has its draft.
    redrafted = graph_of(
        {"draft": DRAFT, "polish": POLISH},
        [["START", "draft", "polish"]],
        [asking("polish", "is_long", {True: "draft", False: "END"})],
    )

    with pytest.raises(WorkflowGraphError, match="stage 'polish' reads"):
        check(looped)
    check(redrafted)


def test_fields_set_before_each_stage_follow_every_path_to_it() -> None:
    graph = graph_of(
        {
            "draft": DRAFT,
            "polish": POLISH,
            "judge": JUDGE,
            "unreached": {**POLISH, "type": "polish"},
        },
        [["START", "draft", "polish", "judge", "END"]],
    )
    writes = {
        "draft": frozenset({"draft"}),
        "polish": frozenset({"answer"}),
        "judge": frozenset({"verdict"}),
        "unreached": frozenset({"answer"}),
    }

    set_before = fields_set_before(graph, writes)

    assert set_before["draft"] == REQUEST_FIELDS
    assert set_before["polish"] == REQUEST_FIELDS | {"draft"}
    assert set_before["judge"] == REQUEST_FIELDS | {"draft", "answer"}
    # A stage no path reaches never runs: nothing it reads can be missing.
    assert set_before["unreached"] == REQUEST_FIELDS | {"draft", "answer", "verdict"}


def test_a_field_the_request_provides_may_be_read_by_the_first_stage() -> None:
    graph = graph_of({"draft": DRAFT, "polish": POLISH}, [["START", "draft", "polish"]])

    check(graph)  # draft reads conversational_query, which the request provides


@pytest.mark.parametrize("name", ["START", "END", "total"])
def test_a_stage_named_like_a_reserved_name_is_refused(name: str) -> None:
    graph = graph_of({name: ANOTHER_DRAFT}, [["START", "END"]])

    with pytest.raises(WorkflowGraphError, match=f"stage '{name}' is named like"):
        check(graph)


def test_a_call_of_a_type_no_declared_component_has_is_refused_saying_so() -> None:
    graph = graph_of({"judge": JUDGE}, [["START", "judge", "END"]])

    with pytest.raises(WorkflowGraphError) as refused:
        check(graph, {"jury": "scribe"})

    assert str(refused.value) == (
        "stage 'judge' calls 'jury', a component of type 'scribe', but its stage "
        "type 'judge' calls one of type 'judge' (expected a component of that "
        "type, and none is declared)"
    )


def test_a_stage_type_with_no_settings_says_it_expects_none() -> None:
    graph = graph_of({"draft": {**DRAFT, "tone": "dry"}}, [["START", "draft"]])

    with pytest.raises(WorkflowGraphError, match=r"tone: .*\(expected no settings\)"):
        check(graph)


@pytest.mark.parametrize(
    ("conditional_edge", "refused"),
    [
        (
            asking("END", "is_long", {True: "END", False: "END"}),
            "conditional edge from 'END' on 'is_long' starts at 'END'",
        ),
        (
            asking("draft", "is_long", {True: "START", False: "END"}),
            "the true answer of the conditional edge from 'draft' on 'is_long' leads "
            "to 'START', which is no stage",
        ),
    ],
)
def test_the_ends_of_the_graph_are_used_only_as_ends(
    conditional_edge: dict[str, Any], refused: str
) -> None:
    graph = graph_of({"draft": DRAFT}, [["START", "draft"]], [conditional_edge])

    with pytest.raises(WorkflowGraphError, match=refused):
        check(graph)


# --- the components before any opens -------------------------------------------------


async def test_the_declared_components_are_the_ones_a_stage_is_given_once_open(
    fake_settings: WorkflowSettings, fake_deps: RagDependencies
) -> None:
    declared = fake_settings.component_types()
    opened = fake_deps.components()

    assert set(declared) == set(opened)
    for name, component in opened.items():
        assert isinstance(component, COMPONENT_TYPES.get(declared[name])), name


def test_the_request_provides_the_fields_of_the_initial_state() -> None:
    assert {
        "conversational_query",
        "standalone_query",
        "conversational_history",
        "mode",
        "metadata_filter",
        "timings_ms",
        "tokens",
    } == REQUEST_FIELDS


# --- plain calls and stated lists ----------------------------------------------------


def test_a_stage_of_no_stage_type_that_states_its_lists_is_a_plain_call() -> None:
    shout = {"call": "voice", "reads": ["conversational_query"], "writes": ["draft"]}
    # polish reads the draft the plain call writes
    graph = graph_of({"shout": shout, "polish": POLISH}, [["START", "shout", "polish"]])

    check(graph, WITH_VOICE)


def test_a_plain_call_whose_lists_do_not_fill_its_request_is_refused() -> None:
    shout = {
        "call": "voice",
        "reads": ["mode", "conversational_history"],
        "writes": ["draft"],
    }
    graph = graph_of({"shout": shout}, [["START", "shout", "END"]])

    with pytest.raises(WorkflowGraphError) as refused:
        check(graph, WITH_VOICE)

    assert str(refused.value) == (
        "stage 'shout': echo takes (text: str) and gives (text: str, length: int); "
        "reads lists 2 fields"
    )


@pytest.mark.parametrize("missing", ["reads", "writes"])
def test_a_stage_without_one_of_its_lists_is_refused_naming_the_stage(
    missing: str,
) -> None:
    shout = {"call": "voice", "reads": ["conversational_query"], "writes": ["draft"]}
    del shout[missing]

    with pytest.raises(ValidationError) as refused:
        graph_of({"shout": shout}, [["START", "shout", "END"]])

    [error] = refused.value.errors()
    assert (error["loc"], error["type"]) == (("stages", "shout", missing), "missing")


def test_a_stage_of_a_stage_type_states_the_lists_its_type_can_use() -> None:
    misstated = {**DRAFT, "reads": ["conversational_query", "standalone_query"]}

    with pytest.raises(WorkflowGraphError) as refused:
        check(graph_of({"draft": misstated}, [["START", "draft", "END"]]))

    assert str(refused.value) == (
        "stage 'draft': the DraftStage reads 1 field, for text: str "
        "(got reads ['conversational_query', 'standalone_query'])"
    )


# --- the types of the fields ---------------------------------------------------------


def test_a_read_of_another_type_is_refused_naming_the_type_expected() -> None:
    misread = {**DRAFT, "reads": ["mode"]}  # draft takes a str message
    graph = graph_of(
        {"draft": misread, "polish": POLISH}, [["START", "draft", "polish"]]
    )

    with pytest.raises(WorkflowGraphError) as refused:
        check(graph)

    assert str(refused.value) == (
        "stage 'draft' reads 'mode' as str, but the request writes it as "
        "Literal['query', 'retrieve'] (expected a field written as str)"
    )


def test_a_read_of_a_field_a_stage_writes_as_another_type_is_refused() -> None:
    # explain takes a verdict, a bool; the draft is a str
    graph = graph_of(
        {"draft": DRAFT, "explain": {**EXPLAIN, "reads": ["draft"]}},
        [["START", "draft", "explain"]],
    )

    with pytest.raises(WorkflowGraphError, match="'explain' reads 'draft' as bool, "):
        check(graph)


def test_two_writers_of_one_field_must_agree() -> None:
    judged = {**JUDGE, "writes": ["draft"]}  # a verdict is a bool
    graph = graph_of(
        {"draft": DRAFT, "judge": judged, "polish": POLISH},
        [["START", "draft", "judge", "polish"]],
    )

    with pytest.raises(
        WorkflowGraphError, match="'draft' is written as str by 'draft' and as bool by"
    ):
        check(graph)


@pytest.mark.parametrize("measured", ["timings_ms", "tokens"])
def test_a_stage_that_writes_a_measured_field_is_refused(measured: str) -> None:
    # The workflow records the measures under each stage's name; a stage's own
    # value would be lost, so a polish that reads it would read no draft.
    drafted = {**DRAFT, "writes": [measured]}
    polished = {**POLISH, "reads": [measured]}
    graph = graph_of(
        {"draft": drafted, "polish": polished}, [["START", "draft", "polish"]]
    )

    with pytest.raises(WorkflowGraphError) as refused:
        check(graph)

    assert str(refused.value) == (
        f"stage 'draft' writes {measured!r}, which the workflow measures for each "
        "stage (expected a field other than 'timings_ms', 'tokens')"
    )


def test_a_field_nothing_reads_is_refused_as_a_likely_typo() -> None:
    typo = {**POLISH, "writes": ["anwser"]}
    graph = graph_of({"draft": DRAFT, "polish": typo}, [["START", "draft", "polish"]])

    with pytest.raises(WorkflowGraphError) as refused:
        check(graph)

    assert str(refused.value) == (
        "stage 'polish' writes 'anwser', but no stage, edge or response reads it, "
        "a likely typo (expected a field a stage reads, or one the response reads: "
        "'answer', 'metadata_filter', 'reranked')"
    )


def test_a_plain_call_reads_fields_of_its_requests_types() -> None:
    shout = {"call": "voice", "reads": ["verdict"], "writes": ["answer"]}
    graph = graph_of(
        {"draft": DRAFT, "judge": JUDGE, "shout": shout},
        [["START", "draft", "judge", "shout"]],
    )

    with pytest.raises(WorkflowGraphError, match="'shout' reads 'verdict' as str"):
        check(graph, WITH_VOICE)
