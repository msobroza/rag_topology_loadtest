"""The fixture stage types, component types and edge conditions the start-up
checks are tested on, in registries of the tests' own.

A draft stage writes a draft from the message, a polish stage turns the draft
into the answer, and a judge stage, which calls a component of another type,
gives a verdict on the draft. A classify stage writes the kind of the message,
a label: a question or a statement. Shared by ``test_graph_check.py`` and
``test_edge_check.py``.

Example::

    check(graph_of({"draft": DRAFT, "polish": POLISH}, [["START", "draft", "polish"]]))
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from echo_components import Echo, EchoRequest
from query_processing.application.edge_conditions.edge_condition import EdgeCondition
from query_processing.application.edge_conditions.label import label
from query_processing.application.graph_check import check_graph
from query_processing.application.stages.base_stage import (
    BaseStage,
    FieldTypes,
    StageContext,
    StageFields,
    StageSettings,
    StageTypes,
    fields_in_order,
)
from query_processing.application.workflow_state import WorkflowState
from query_processing.domain.base_component import BaseComponent
from query_processing.domain.graph_description import GraphDescription
from query_processing.domain.registry import Registry

FIXTURE_COMPONENT_TYPES: Registry[type[BaseComponent[Any, Any]]] = Registry(
    "component type"
)


class Draft(BaseModel):
    # The two ports' request and answer: a component type's are pydantic
    # models, though the fixture stage types declare the fields they use.
    text: str


class FakeScribe(
    BaseComponent[Draft, Draft],
    component_type="scribe",
    registry=FIXTURE_COMPONENT_TYPES,
):
    """The port a draft and a polish stage call."""


class FakeJudge(
    BaseComponent[Draft, Draft],
    component_type="judge",
    registry=FIXTURE_COMPONENT_TYPES,
):
    """The port a judge stage calls."""


class FakeEcho(
    BaseComponent[EchoRequest, Echo],
    component_type="echo",
    registry=FIXTURE_COMPONENT_TYPES,
):
    """The port a plain call of these graphs calls."""


class NoSettings(StageSettings):
    pass


class FixtureStage(BaseStage[Any, NoSettings]):
    """A stage type that reads one field and writes one, whatever their names."""

    settings_type = NoSettings
    takes: Any = str  # the type of the field it reads
    gives: Any = str  # the type of the field it writes
    labels: tuple[str, ...] = ()  # the labels of the field it writes, if any

    @classmethod
    def check_fields(
        cls, settings: NoSettings, fields: StageFields, context: StageContext
    ) -> FieldTypes:
        expecting = f"the {cls.__name__}"
        labels = {fields.writes[0]: cls.labels} if cls.labels else {}
        return FieldTypes(
            fields_in_order("reads", fields.reads, {"text": cls.takes}, expecting),
            fields_in_order("writes", fields.writes, {"text": cls.gives}, expecting),
            labels,
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


class ClassifyStage(FixtureStage):
    component_type = FakeScribe
    labels = ("question", "statement")


FIXTURE_STAGE_TYPES: StageTypes = Registry("stage type")
for _name, _stage_type in [
    ("draft", DraftStage),
    ("polish", PolishStage),
    ("judge", JudgeStage),
    ("explain", ExplainStage),
    ("classify", ClassifyStage),
]:
    FIXTURE_STAGE_TYPES.register(_name)(_stage_type)
FIXTURE_EDGE_CONDITIONS: Registry[EdgeCondition] = Registry("edge condition")
FIXTURE_EDGE_CONDITIONS.register("label")(label)
COMPONENTS = {"pen": "scribe", "jury": "judge"}
WITH_VOICE = {**COMPONENTS, "voice": FakeEcho.component_type_name}


@FIXTURE_EDGE_CONDITIONS.register("is_question")
def is_question(conversational_query: str) -> bool:
    return conversational_query.endswith("?")


@FIXTURE_EDGE_CONDITIONS.register("is_long")
def is_long(conversational_query: str) -> bool:
    return len(conversational_query) > 10


def graph_of(
    stages: dict[str, Any],
    edges: list[list[str]],
    conditional_edges: list[dict[str, Any]] | None = None,
) -> GraphDescription:
    """A graph of these stages and edges."""
    return GraphDescription.model_validate(
        {"stages": stages, "edges": edges, "conditional_edges": conditional_edges or []}
    )


def check(graph: GraphDescription, components: dict[str, str] = COMPONENTS) -> None:
    """``check_graph`` on the fixture registries."""
    check_graph(
        graph,
        components,
        StageContext(),
        component_types=FIXTURE_COMPONENT_TYPES,
        stage_types=FIXTURE_STAGE_TYPES,
        edge_conditions=FIXTURE_EDGE_CONDITIONS,
    )


def asking(source: str, path: str, path_map: dict[Any, str]) -> dict[str, Any]:
    """A conditional edge of a fixture yes-or-no edge condition, which reads the message."""
    return {
        "source": source,
        "path": path,
        "reads": ["conversational_query"],
        "path_map": path_map,
    }


DRAFT = {"call": "pen", "reads": ["conversational_query"], "writes": ["draft"]}
POLISH = {"call": "pen", "reads": ["draft"], "writes": ["answer"]}
JUDGE = {"call": "jury", "reads": ["draft"], "writes": ["verdict"]}
EXPLAIN = {"call": "pen", "reads": ["verdict"], "writes": ["answer"]}
CLASSIFY = {"call": "pen", "reads": ["conversational_query"], "writes": ["kind"]}
ANOTHER_DRAFT = {**DRAFT, "type": "draft"}
