"""The start-up checks are generic: graphs of fixture stage types, fixture component
types and fixture edge conditions, in registries of the test's own.

A draft stage writes a draft from the message, a polish stage turns the draft
into the answer, and a judge stage, which calls a component of another type,
gives a verdict on the draft. The RAG's own graphs are checked through
``qp-serve`` in ``tests/entrypoints/test_wrong_graph_at_start_up.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from echo_components import Echo, EchoRequest
from query_processing.application.component_types import COMPONENT_TYPES
from query_processing.application.edge_conditions.edge_condition import EdgeCondition
from query_processing.application.graph_check import (
    WorkflowGraphError,
    check_graph,
    fields_set_before,
)
from query_processing.application.rag_dependencies import RagDependencies
from query_processing.application.stages.base_stage import (
    BaseStage,
    FieldTypes,
    StageFields,
    StageSettings,
    StageTypes,
    fields_in_order,
)
from query_processing.application.workflow_state import REQUEST_FIELDS, WorkflowState
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.base_component import BaseComponent
from query_processing.domain.graph_description import GraphDescription
from query_processing.domain.registry import Registry

FIXTURE_COMPONENT_TYPES: Registry[type[BaseComponent[Any, Any]]] = Registry(
    "component type"
)


class FakeScribe(
    BaseComponent[str, str], component_type="scribe", registry=FIXTURE_COMPONENT_TYPES
):
    """The port a draft and a polish stage call."""


class FakeJudge(
    BaseComponent[str, bool], component_type="judge", registry=FIXTURE_COMPONENT_TYPES
):
    """The port a judge stage calls."""


class NoSettings(StageSettings):
    pass


class FixtureStage(BaseStage[Any, NoSettings]):
    """A stage type that reads one field and writes one, whatever their names."""

    settings_type = NoSettings
    takes: Any = str  # the type of the field it reads
    gives: Any = str  # the type of the field it writes

    @classmethod
    def check_fields(cls, settings: NoSettings, fields: StageFields) -> FieldTypes:
        expecting = f"the {cls.__name__}"
        return FieldTypes(
            fields_in_order("reads", fields.reads, {"text": cls.takes}, expecting),
            fields_in_order("writes", fields.writes, {"text": cls.gives}, expecting),
        )

    async def run(self, state: WorkflowState) -> WorkflowState:
        return {}


class DraftStage(FixtureStage):
    component_type = FakeScribe


class PolishStage(FixtureStage):
    component_type = FakeScribe


class JudgeStage(FixtureStage):
    component_type = FakeJudge
    gives = bool  # a verdict


class ExplainStage(FixtureStage):
    component_type = FakeScribe
    takes = bool  # a verdict, explained in words


FIXTURE_STAGE_TYPES: StageTypes = Registry("stage type")
for _name, _stage_type in [
    ("draft", DraftStage),
    ("polish", PolishStage),
    ("judge", JudgeStage),
    ("explain", ExplainStage),
]:
    FIXTURE_STAGE_TYPES.register(_name)(_stage_type)
FIXTURE_EDGE_CONDITIONS: Registry[EdgeCondition] = Registry("edge condition")
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


DRAFT = {"call": "pen", "reads": ["conversational_query"], "writes": ["draft"]}
POLISH = {"call": "pen", "reads": ["draft"], "writes": ["answer"]}
JUDGE = {"call": "jury", "reads": ["draft"], "writes": ["verdict"]}
EXPLAIN = {"call": "pen", "reads": ["verdict"], "writes": ["answer"]}
ANOTHER_DRAFT = {**DRAFT, "type": "draft"}


def test_a_graph_of_other_stage_types_and_edge_conditions_passes() -> None:
    graph = graph_of(
        {"draft": DRAFT, "polish": POLISH, "judge": JUDGE, "explain": EXPLAIN},
        [["START", "draft", "judge", "explain"], ["polish", "END"]],
        [
            {
                "source": "explain",
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
        "history",
        "mode",
        "timings_ms",
    } == REQUEST_FIELDS


# --- plain calls and stated lists ----------------------------------------------------


class FakeEcho(
    BaseComponent[EchoRequest, Echo],
    component_type="echo",
    registry=FIXTURE_COMPONENT_TYPES,
):
    """The port a plain call of these graphs calls."""


WITH_VOICE = {**COMPONENTS, "voice": FakeEcho.component_type_name}


def test_a_stage_of_no_stage_type_that_states_its_lists_is_a_plain_call() -> None:
    shout = {"call": "voice", "reads": ["conversational_query"], "writes": ["draft"]}
    # polish reads the draft the plain call writes
    graph = graph_of({"shout": shout, "polish": POLISH}, [["START", "shout", "polish"]])

    check(graph, WITH_VOICE)


def test_a_plain_call_whose_lists_do_not_fill_its_request_is_refused() -> None:
    shout = {"call": "voice", "reads": ["mode", "history"], "writes": ["draft"]}
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


def test_a_field_nothing_reads_is_refused_as_a_likely_typo() -> None:
    typo = {**POLISH, "writes": ["anwser"]}
    graph = graph_of({"draft": DRAFT, "polish": typo}, [["START", "draft", "polish"]])

    with pytest.raises(WorkflowGraphError) as refused:
        check(graph)

    assert str(refused.value) == (
        "stage 'polish' writes 'anwser', but no stage, edge or response reads it, "
        "a likely typo (expected a field a stage reads, or one the response reads: "
        "'answer', 'reranked')"
    )


def test_a_plain_call_reads_fields_of_its_requests_types() -> None:
    shout = {"call": "voice", "reads": ["verdict"], "writes": ["answer"]}
    graph = graph_of(
        {"draft": DRAFT, "judge": JUDGE, "shout": shout},
        [["START", "draft", "judge", "shout"]],
    )

    with pytest.raises(WorkflowGraphError, match="'shout' reads 'verdict' as str"):
        check(graph, WITH_VOICE)
