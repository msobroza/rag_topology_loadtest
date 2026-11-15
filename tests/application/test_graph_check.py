"""The start-up checks are generic: graphs of fixture stage types, fixture component
types and fixture edge conditions, in registries of the test's own.

A draft stage writes a draft from the message, a polish stage turns the draft
into the answer, and a judge stage, which calls a component of another type,
gives a verdict on the draft. The RAG's own graphs are checked through
``rag-serve`` in ``tests/entrypoints/test_wrong_graph_at_start_up.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_load_test.application.component_types import COMPONENT_TYPES
from rag_load_test.application.edge_conditions.edge_condition import EdgeCondition
from rag_load_test.application.graph_check import (
    WorkflowGraphError,
    check_graph,
    fields_set_before,
)
from rag_load_test.application.rag_dependencies import RagDependencies
from rag_load_test.application.stages.base_stage import (
    BaseStage,
    StageSettings,
    StageTypes,
)
from rag_load_test.application.workflow_state import REQUEST_FIELDS, WorkflowState
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.graph_description import GraphDescription
from rag_load_test.domain.registry import Registry


class FakeScribe:
    """The port a draft and a polish stage call."""


class FakeJudge:
    """The port a judge stage calls."""


class NoSettings(StageSettings):
    pass


class FixtureStage(BaseStage[Any, NoSettings]):
    settings_type = NoSettings

    async def run(self, state: WorkflowState) -> WorkflowState:
        return {}


class DraftStage(FixtureStage):
    component_type = FakeScribe
    reads = frozenset({"conversational_query"})
    writes = frozenset({"draft"})


class PolishStage(FixtureStage):
    component_type = FakeScribe
    reads = frozenset({"draft"})
    writes = frozenset({"answer"})


class JudgeStage(FixtureStage):
    component_type = FakeJudge
    reads = frozenset({"draft"})
    writes = frozenset({"verdict"})


FIXTURE_STAGE_TYPES: StageTypes = Registry("stage type")
for _name, _stage_type in [
    ("draft", DraftStage),
    ("polish", PolishStage),
    ("judge", JudgeStage),
]:
    FIXTURE_STAGE_TYPES.register(_name)(_stage_type)
FIXTURE_EDGE_CONDITIONS: Registry[EdgeCondition] = Registry("edge condition")
FIXTURE_COMPONENT_TYPES = {"scribe": FakeScribe, "judge": FakeJudge}
COMPONENTS = {"pen": "scribe", "jury": "judge"}


@FIXTURE_EDGE_CONDITIONS.register("is_question")
def is_question(state: WorkflowState) -> bool:
    return state.get("conversational_query", "").endswith("?")


@FIXTURE_EDGE_CONDITIONS.register("is_long")
def is_long(state: WorkflowState) -> bool:
    return len(state.get("conversational_query", "")) > 10


def graph_of(
    stages: dict[str, Any],
    edges: list[list[str]],
    conditional_edges: list[dict[str, Any]] | None = None,
) -> GraphDescription:
    return GraphDescription.model_validate(
        {"stages": stages, "edges": edges, "conditional_edges": conditional_edges or []}
    )


def check(graph: GraphDescription, components: dict[str, str] = COMPONENTS) -> None:
    check_graph(
        graph,
        components,
        component_types=FIXTURE_COMPONENT_TYPES,
        stage_types=FIXTURE_STAGE_TYPES,
        edge_conditions=FIXTURE_EDGE_CONDITIONS,
    )


DRAFT = {"call": "pen"}
POLISH = {"call": "pen"}
JUDGE = {"call": "jury"}
ANOTHER_DRAFT = {"type": "draft", "call": "pen"}


def test_a_graph_of_other_stage_types_and_edge_conditions_passes() -> None:
    graph = graph_of(
        {"draft": DRAFT, "polish": POLISH, "judge": JUDGE},
        [["START", "draft", "judge"], ["polish", "END"]],
        [
            {
                "source": "judge",
                "path": "is_question",
                "path_map": {True: "polish", False: "END"},
            },
            {
                "source": "draft",
                "path": "is_question",  # the same edge condition on another source
                "path_map": {True: "END", False: "END"},
            },
        ],
    )

    check(graph)


def test_a_field_written_on_one_branch_only_is_refused() -> None:
    graph = graph_of(
        {"draft": DRAFT, "polish": POLISH},
        [["draft", "polish", "END"]],
        [
            {
                "source": "START",
                "path": "is_question",
                "path_map": {True: "draft", False: "polish"},
            }
        ],
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
        [
            {
                "source": "START",
                "path": "is_long",
                "path_map": {True: "long", False: "short"},
            }
        ],
    )

    check(graph)


def test_a_stage_on_a_loop_reads_only_what_is_set_before_its_first_pass() -> None:
    # START -> polish -> draft -> polish ...: the first polish has no draft.
    looped = graph_of(
        {"draft": DRAFT, "polish": POLISH},
        [["START", "polish", "draft"]],
        [
            {
                "source": "draft",
                "path": "is_long",
                "path_map": {True: "polish", False: "END"},
            }
        ],
    )
    # START -> draft -> polish -> draft ...: every pass of polish has its draft.
    redrafted = graph_of(
        {"draft": DRAFT, "polish": POLISH},
        [["START", "draft", "polish"]],
        [
            {
                "source": "polish",
                "path": "is_long",
                "path_map": {True: "draft", False: "END"},
            }
        ],
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
            "unreached": {"type": "polish", "call": "pen"},
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
    graph = graph_of({"draft": DRAFT}, [["START", "draft", "END"]])

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
    graph = graph_of({"draft": {"call": "pen", "tone": "dry"}}, [["START", "draft"]])

    with pytest.raises(WorkflowGraphError, match=r"tone: .*\(expected no settings\)"):
        check(graph)


@pytest.mark.parametrize(
    ("conditional_edge", "refused"),
    [
        (
            {
                "source": "END",
                "path": "is_long",
                "path_map": {True: "END", False: "END"},
            },
            "conditional edge from 'END' on 'is_long' starts at 'END'",
        ),
        (
            {
                "source": "draft",
                "path": "is_long",
                "path_map": {True: "START", False: "END"},
            },
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
    fake_settings: RagSettings, fake_deps: RagDependencies
) -> None:
    declared = fake_settings.component_types()
    opened = fake_deps.components()

    assert set(declared) == set(opened)
    for name, component in opened.items():
        assert isinstance(component, COMPONENT_TYPES[declared[name]]), name


def test_the_request_provides_the_fields_of_the_initial_state() -> None:
    assert {
        "conversational_query",
        "standalone_query",
        "history",
        "mode",
        "timings_ms",
    } == REQUEST_FIELDS
