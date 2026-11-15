"""``build_workflow``: a graph description and its components in, a workflow out.

The builder knows no stage, no component and no edge condition. Each stage is an
instance of the stage type its description names, built with the component its
``call`` names and its own settings, and timed under its name. Each chain of
plain edges becomes one LangGraph edge per pair, the reserved ``START`` and
``END`` becoming LangGraph's own. Each conditional edge asks the edge condition
its ``path`` names, handed to LangGraph as it is registered: LangGraph names the
branch after the function, and Domino tracing needs that name.

Example:
    >>> workflow = build_workflow(settings.workflow_graph(), deps.components())
    >>> result = await workflow.run("How many vacation days?")
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Hashable, Mapping
from typing import Any, Protocol

from langgraph.graph import END as LANGGRAPH_END
from langgraph.graph import START as LANGGRAPH_START
from langgraph.graph import StateGraph

from ..domain.graph_description import (
    END,
    START,
    ConditionalEdge,
    GraphDescription,
    StageDescription,
)
from ..domain.registry import Registry
from .edge_conditions.edge_condition import EDGE_CONDITIONS, EdgeCondition
from .stages.base_stage import STAGE_TYPES, BaseStage
from .workflow import CompiledWorkflow
from .workflow_state import WorkflowState

StageTypes = Registry[type[BaseStage[Any, Any]]]

# The description's reserved names, as LangGraph spells them.
_LANGGRAPH_NAMES = {START: LANGGRAPH_START, END: LANGGRAPH_END}


class LangGraphNode(Protocol):
    """What LangGraph runs for one stage (LangGraph calls it a node).

    A Protocol rather than ``Callable[[WorkflowState], ...]`` because LangGraph
    1.x types a node by a parameter named ``state``; a ``Callable`` alias erases
    the name and mypy then rejects ``StateGraph.add_node``.
    """

    def __call__(self, state: WorkflowState) -> Awaitable[WorkflowState]:
        """Run the stage.

        Args:
            state: The workflow state so far.

        Returns:
            An awaitable of the stage's update.
        """


def build_workflow(
    graph: GraphDescription,
    components: Mapping[str, object],
    *,
    stage_types: StageTypes = STAGE_TYPES,
    edge_conditions: Registry[EdgeCondition] = EDGE_CONDITIONS,
) -> CompiledWorkflow:
    """Build and compile the graph a description gives.

    Args:
        graph: The stages, the chains of plain edges and the conditional edges.
        components: What a stage's ``call`` names, by name.
        stage_types: Where a stage's type is found; the registered stage types
            unless a test hands in its own.
        edge_conditions: Where a conditional edge's ``path`` is found; the
            registered edge conditions unless a test hands in its own.

    Returns:
        The compiled workflow.

    Raises:
        ValueError: A stage type or an edge condition is not registered (the
            registry names the known ones), a stage's settings do not fit its
            stage type, or a stage type refuses its component.

    Example:
        >>> build_workflow(settings.workflow_graph(), deps.components())
    """
    # ty 0.0.100 does not match a TypedDict class against LangGraph's
    # TypedDictLike protocol (its __required_keys__); mypy accepts it.
    builder = StateGraph(WorkflowState)  # ty: ignore[invalid-argument-type]
    for name, description in graph.stages.items():
        stage = open_stage(name, description, components, stage_types)
        builder.add_node(name, timed(name, stage))
    for source, target in graph.edge_pairs():
        builder.add_edge(langgraph_name(source), langgraph_name(target))
    for edge in graph.conditional_edges:
        condition = edge_conditions.get(edge.path)
        source = langgraph_name(edge.source)
        builder.add_conditional_edges(source, condition, path_map(edge))
    return CompiledWorkflow(builder.compile())


def open_stage(
    name: str,
    description: StageDescription,
    components: Mapping[str, object],
    stage_types: StageTypes,
) -> BaseStage[Any, Any]:
    """Build one stage: its stage type, with its settings and its component.

    Args:
        name: The stage's name, which its type defaults to.
        description: The stage's type, ``call`` and own settings.
        components: What ``call`` names, by name.
        stage_types: Where the stage's type is found.

    Returns:
        The stage.

    Example:
        >>> open_stage("rerank", StageDescription.model_validate(
        ...     {"call": "reranker", "top_k": 10}), components, STAGE_TYPES)
        <RerankStage ...>
    """
    stage_type = stage_types.get(description.type_name(name))
    settings = stage_type.settings_type.model_validate(description.stage_settings)
    return stage_type(components[description.call], settings)


def timed(name: str, stage: BaseStage[Any, Any]) -> LangGraphNode:
    """Wrap a stage so its update also carries ``{"timings_ms": {name: ms}}``.

    Args:
        name: The stage's name, the key its wall time is recorded under.
        stage: The stage to time.

    Returns:
        What LangGraph runs for the stage.

    Example:
        >>> builder.add_node("rerank", timed("rerank", stage))
    """

    async def run(state: WorkflowState) -> WorkflowState:
        started = time.perf_counter()
        update = await stage.run(state)
        return {**update, "timings_ms": {name: (time.perf_counter() - started) * 1000}}

    return run


def langgraph_name(name: str) -> str:
    """Return a stage name as LangGraph knows it: the reserved names its own way.

    Args:
        name: A stage name, ``START`` or ``END``.

    Returns:
        LangGraph's start or end for the reserved names, else ``name``.

    Example:
        >>> langgraph_name("END")
        '__end__'
    """
    return _LANGGRAPH_NAMES.get(name, name)


def path_map(edge: ConditionalEdge) -> dict[Hashable, str]:
    """Return a conditional edge's ``path_map`` with LangGraph's names.

    Args:
        edge: The conditional edge.

    Returns:
        Each answer of the edge condition, mapped to where it leads.

    Example:
        >>> path_map(ConditionalEdge(source="rerank", path="is_query_mode",
        ...          path_map={True: "generate", False: "END"}))
        {True: 'generate', False: '__end__'}
    """
    return {answer: langgraph_name(to) for answer, to in edge.path_map.items()}
