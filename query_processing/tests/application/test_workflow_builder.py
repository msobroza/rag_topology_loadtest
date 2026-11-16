"""The builder is generic: a graph of another shape, from fixture stage types and a
fixture edge condition registered in registries of the test's own.

The fixture graph drafts an answer by copying the message with a suffix, then
shouts it only when the message is a question:

    START -> draft -?-> shout -> END
                   \\-> END
"""

from __future__ import annotations

from typing import Any

import pytest

from echo_components import Upper
from query_processing.application.edge_conditions.edge_condition import EdgeCondition
from query_processing.application.graph_check import WorkflowGraphError
from query_processing.application.stages.base_stage import BaseStage, StageSettings
from query_processing.application.stages.call_stage import CallStage
from query_processing.application.workflow import CompiledWorkflow
from query_processing.application.workflow_builder import (
    build_workflow,
    langgraph_name,
    open_stage,
    path_map,
    timed,
)
from query_processing.application.workflow_state import WorkflowState, initial_state
from query_processing.domain.graph_description import (
    ConditionalEdge,
    GraphDescription,
    StageDescription,
)
from query_processing.domain.registry import Registry
from workflow_fakes import ChainNameRecorder


class FakeScribe:
    """A component of the fixture graph: writes back what it is given, and records it."""

    def __init__(self) -> None:
        self.written: list[str] = []

    async def write(self, text: str) -> str:
        self.written.append(text)
        return text


FIXTURE_STAGE_TYPES: Registry[type[BaseStage[Any, Any]]] = Registry("stage type")
FIXTURE_EDGE_CONDITIONS: Registry[EdgeCondition] = Registry("edge condition")


class SuffixSettings(StageSettings):
    suffix: str


class NoSettings(StageSettings):
    pass


@FIXTURE_STAGE_TYPES.register("copy")
class CopyStage(BaseStage[FakeScribe, SuffixSettings]):
    component_type = FakeScribe
    settings_type = SuffixSettings
    reads = frozenset({"conversational_query"})
    writes = frozenset({"answer"})

    async def run(self, state: WorkflowState) -> WorkflowState:
        text = state["conversational_query"] + self.settings.suffix
        return {"answer": await self.component.write(text)}


@FIXTURE_STAGE_TYPES.register("shout")
class ShoutStage(BaseStage[FakeScribe, NoSettings]):
    component_type = FakeScribe
    settings_type = NoSettings
    reads = frozenset({"answer"})
    writes = frozenset({"answer"})

    async def run(self, state: WorkflowState) -> WorkflowState:
        return {"answer": await self.component.write(str(state["answer"]).upper())}


@FIXTURE_EDGE_CONDITIONS.register("is_a_question")
def is_a_question(state: WorkflowState) -> bool:
    return state["conversational_query"].endswith("?")


FIXTURE_GRAPH = {
    "stages": {
        "draft": {"type": "copy", "call": "scribe", "suffix": "!"},
        "shout": {"call": "scribe"},  # its type is its name
    },
    "edges": [["START", "draft"], ["shout", "END"]],
    "conditional_edges": [
        {
            "source": "draft",
            "path": "is_a_question",
            "path_map": {True: "shout", False: "END"},
        }
    ],
}


def fixture_workflow(scribe: FakeScribe, **changes: Any) -> CompiledWorkflow:
    graph = GraphDescription.model_validate({**FIXTURE_GRAPH, **changes})
    return build_workflow(
        graph,
        {"scribe": scribe},
        stage_types=FIXTURE_STAGE_TYPES,
        edge_conditions=FIXTURE_EDGE_CONDITIONS,
    )


@pytest.mark.parametrize(
    ("message", "answer", "written"),
    [
        ("who?", "WHO?!", ["who?!", "WHO?!"]),  # a question: draft, then shout
        ("hello", "hello!", ["hello!"]),  # not one: draft, then the end
    ],
)
async def test_a_graph_of_another_shape_takes_the_branch_its_edge_condition_picks(
    message: str, answer: str, written: list[str]
) -> None:
    scribe = FakeScribe()

    result = await fixture_workflow(scribe).run(message)

    assert result.answer == answer
    assert scribe.written == written


@pytest.mark.parametrize(
    ("message", "timed_stages"), [("who?", {"draft", "shout"}), ("hello", {"draft"})]
)
async def test_each_stage_that_runs_is_timed_under_its_own_name(
    message: str, timed_stages: set[str]
) -> None:
    graph = fixture_workflow(FakeScribe()).graph

    final = await graph.ainvoke(initial_state(message, "query"))

    assert set(final["timings_ms"]) == timed_stages
    assert all(ms > 0 for ms in final["timings_ms"].values())


@pytest.mark.parametrize(
    ("message", "skipped"), [("who?", set[str]()), ("hello", {"shout"})]
)
async def test_the_result_times_every_stage_of_the_graph_by_name_then_the_total(
    message: str, skipped: set[str]
) -> None:
    result = await fixture_workflow(FakeScribe()).run(message)

    assert list(result.timings_ms) == ["draft", "shout", "total"]
    zero = {stage for stage, ms in result.timings_ms.items() if ms == 0.0}
    assert zero == skipped  # a stage that did not run keeps its key, at 0


async def test_the_result_lists_the_stages_in_the_order_the_description_gives() -> None:
    stages = {"shout": {"call": "scribe"}, **FIXTURE_GRAPH["stages"]}

    result = await fixture_workflow(FakeScribe(), stages=stages).run("who?")

    assert list(result.timings_ms) == ["shout", "draft", "total"]


async def test_every_step_reaches_a_tracer_with_a_name() -> None:
    recorder = ChainNameRecorder()
    graph = fixture_workflow(FakeScribe()).graph

    await graph.ainvoke(initial_state("who?", "query"), {"callbacks": [recorder]})

    assert None not in recorder.names, recorder.names
    # The edge condition keeps its own function name: the branch is named after it.
    assert {"draft", "is_a_question", "shout"} <= set(recorder.names)


def test_a_longer_chain_is_one_edge_per_pair() -> None:
    chained = [["START", "draft", "shout", "END"]]

    workflow = fixture_workflow(FakeScribe(), edges=chained, conditional_edges=[])

    edges = {(edge.source, edge.target) for edge in workflow.graph.get_graph().edges}
    assert edges == {("__start__", "draft"), ("draft", "shout"), ("shout", "__end__")}


def test_a_setting_its_stage_type_does_not_know_is_refused_naming_it() -> None:
    misspelt = {"type": "copy", "call": "scribe", "suffix": "!", "sufix": "!"}
    stages = {**FIXTURE_GRAPH["stages"], "draft": misspelt}

    with pytest.raises(ValueError, match="sufix"):
        fixture_workflow(FakeScribe(), stages=stages)


async def test_a_plain_call_in_the_graph_answers_through_its_component() -> None:
    # "loud" is no stage type: it states its lists, so it is a plain call.
    loud = {"call": "echo", "reads": ["answer"], "writes": ["answer"]}
    stages = {**FIXTURE_GRAPH["stages"], "loud": loud}
    graph = GraphDescription.model_validate(
        {"stages": stages, "edges": [["START", "draft", "loud", "END"]]}
    )
    workflow = build_workflow(
        graph,
        {"scribe": FakeScribe(), "echo": Upper()},
        stage_types=FIXTURE_STAGE_TYPES,
        edge_conditions=FIXTURE_EDGE_CONDITIONS,
    )

    result = await workflow.run("hello")

    assert result.answer == "HELLO!"
    assert list(result.timings_ms) == ["draft", "shout", "loud", "total"]


def test_a_plain_call_of_something_that_answers_no_ainvoke_is_refused() -> None:
    loud = {"call": "scribe", "reads": ["answer"], "writes": ["answer"]}

    with pytest.raises(WorkflowGraphError, match="stage 'loud': a plain call calls"):
        fixture_workflow(FakeScribe(), stages={**FIXTURE_GRAPH["stages"], "loud": loud})


# --- the builder's pieces -------------------------------------------------------


def test_open_stage_builds_a_plain_call_bound_by_its_lists() -> None:
    description = StageDescription.model_validate(
        {"call": "echo", "reads": ["conversational_query"], "writes": ["answer"]}
    )

    stage = open_stage("loud", description, {"echo": Upper()}, FIXTURE_STAGE_TYPES)

    assert isinstance(stage, CallStage)
    assert stage.binding.from_reads == {"text": "conversational_query"}


def test_open_stage_builds_the_stage_type_its_description_names() -> None:
    scribe = FakeScribe()
    description = StageDescription.model_validate(
        {"type": "copy", "call": "scribe", "suffix": "?"}
    )

    stage = open_stage("draft", description, {"scribe": scribe}, FIXTURE_STAGE_TYPES)

    assert isinstance(stage, CopyStage)
    assert (stage.component, stage.settings) == (scribe, SuffixSettings(suffix="?"))


async def test_timed_adds_the_stage_wall_time_under_its_name_to_its_update() -> None:
    stage = ShoutStage(FakeScribe(), NoSettings())

    update = await timed("loud", stage)({"answer": "hi"})

    assert update["answer"] == "HI"
    assert set(update["timings_ms"]) == {"loud"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [("START", "__start__"), ("END", "__end__"), ("draft", "draft")],
)
def test_langgraph_name_spells_the_reserved_names_langgraph_s_way(
    name: str, expected: str
) -> None:
    assert langgraph_name(name) == expected


def test_path_map_leads_each_answer_where_langgraph_knows_the_target() -> None:
    edge = ConditionalEdge.model_validate(FIXTURE_GRAPH["conditional_edges"][0])

    assert path_map(edge) == {True: "shout", False: "__end__"}
