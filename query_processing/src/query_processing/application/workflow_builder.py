"""``build_workflow``: a graph description and its components in, a workflow out.

The builder knows no stage, no component and no edge condition. Each stage is an
instance of the stage type its description names, built with the component its
``call`` names, its own settings and its ``reads`` and ``writes``, or, when it is
of no stage type, a plain call bound by them; each is timed under its name; the
workflow's
results are keyed by those names, in the description's order. The state's
fields are those the request and the stages write. Each conditional edge asks
the edge condition its ``path`` names about the values of the fields its
``reads`` names, through a function that keeps the edge condition's name: an
orchestrator names the branch after it, and Domino tracing needs that name.
The answer, a bool or a label, picks its row of ``path_map``. All of it is
handed, as the opened graph, to the orchestrator's class, which links the
stages with its library (ADR 0010).

Example:
    >>> workflow = build_workflow(settings.graph, deps.components(),
    ...     stage_context(settings), composition.orchestrator())
    >>> result = await workflow.run("How many vacation days?")
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from ..adapters.orchestrator.base_orchestrator import BaseOrchestrator
from ..domain.components.base_component import BaseComponent
from ..domain.graph_description import (
    ConditionalEdge,
    GraphDescription,
    StageDescription,
)
from ..domain.opened_graph import OpenedConditionalEdge, OpenedGraph, StageRun
from ..domain.registry import Registry
from .edge_check import edge_name
from .edge_conditions.edge_condition import EDGE_CONDITIONS, EdgeCondition
from .graph_check import WorkflowGraphError
from .stage_binding import bind_call, stage_fields, stage_type_of
from .stages.base_stage import STAGE_TYPES, BaseStage, StageContext, StageTypes
from .stages.call_stage import CallStage
from .state_fields import STATE_REDUCERS, field_types_of, state_field_types
from .workflow import CompiledWorkflow
from .workflow_state import TIMINGS_FIELD, TOKENS_FIELD, WorkflowState


def build_workflow(
    graph: GraphDescription,
    components: Mapping[str, object],
    context: StageContext,
    orchestrator: type[BaseOrchestrator],
    *,
    stage_types: StageTypes = STAGE_TYPES,
    edge_conditions: Registry[EdgeCondition] = EDGE_CONDITIONS,
) -> CompiledWorkflow:
    """Build the graph a description gives, on the orchestrator handed in.

    Args:
        graph: The stages, the chains of plain edges and the conditional edges.
        components: What a stage's ``call`` names, by name.
        context: What every stage of a stage type is given, such as the prompts.
        orchestrator: The library that runs the graph, the class the
            composition found under the settings' ``orchestrator``.
        stage_types: Where a stage's type is found; the registered stage types
            unless a test hands in its own.
        edge_conditions: Where a conditional edge's ``path`` is found; the
            registered edge conditions unless a test hands in its own.

    Returns:
        The compiled workflow.

    Raises:
        ValueError: A stage type or an edge condition is not registered (the
            registry names the known ones); ``check_graph`` refuses that first,
            before any component opens.
        WorkflowGraphError: A stage's settings do not fit its stage type, a
            stage type refuses its opened component, such as a store that ranks
            no text under a lexical retrieval, or two writers write a field as
            two types.

    Example:
        >>> build_workflow(settings.graph, deps.components(),
        ...                stage_context(settings), composition.orchestrator())
    """
    stages = {
        name: open_stage(name, description, components, stage_types, context)
        for name, description in graph.stages.items()
    }
    types = field_types_of({name: stage.field_types for name, stage in stages.items()})
    asked = [
        OpenedConditionalEdge(e, asked_of_state(edge_conditions.get(e.path), e))
        for e in graph.conditional_edges
    ]
    opened = OpenedGraph(
        stages={name: timed(name, stage) for name, stage in stages.items()},
        edges=graph.edge_pairs(),
        conditional_edges=asked,
        field_types=state_field_types(types),
        reducers=STATE_REDUCERS,
    )
    return CompiledWorkflow(orchestrator(opened), stage_names=list(graph.stages))


def asked_of_state(
    condition: EdgeCondition, edge: ConditionalEdge
) -> Callable[[WorkflowState], bool | str]:
    """Return the edge condition asked of the state, its answer checked for a row.

    The condition is handed the values of the edge's ``reads``, in order. The
    function keeps the edge condition's name: the orchestrator names the
    branch after it, and Domino tracing fails on a step with no name. An answer
    the edge has no row for is refused here: LangGraph would fail on it with a
    bare ``KeyError``.

    Args:
        condition: The registered edge condition.
        edge: Its conditional edge: the fields it reads, in the order the
            condition takes them, and the answers it has a row for.

    Returns:
        A function of the workflow state, named as ``condition`` is.

    Raises:
        WorkflowGraphError: When asked, the condition answers what the edge
            has no row for (see ``answer_with_a_row``).

    Example:
        >>> asked_of_state(is_query_mode, edge)({"mode": "retrieve"})
        False
    """
    fields = tuple(edge.reads)

    def ask(state: WorkflowState) -> bool | str:
        return answer_with_a_row(edge, condition(*(state[field] for field in fields)))

    ask.__name__ = ask.__qualname__ = getattr(condition, "__name__", "edge_condition")
    return ask


def answer_with_a_row(edge: ConditionalEdge, answer: object) -> bool | str:
    """Return an edge condition's answer when its edge has a row for it.

    Args:
        edge: The conditional edge that asked.
        answer: What its edge condition answered.

    Returns:
        ``answer``, a key of the edge's ``path_map``.

    Raises:
        WorkflowGraphError: The edge has no row for ``answer``: a fault of the
            use case's edge condition, not of a model or a service. The message
            names the edge's source, the condition, the answer and its type,
            and the answers the edge has a row for.

    Example:
        >>> answer_with_a_row(edge, True)
        True
    """
    # A bool or a str only: 1 is equal to True, and would take True's row.
    if isinstance(answer, bool | str) and answer in edge.path_map:
        return answer
    raise WorkflowGraphError(
        f"{edge_name(edge)}: the edge condition answered {answer!r} "
        f"({type(answer).__name__}), which the edge has no row for (expected "
        f"one of {list(edge.path_map)!r})"
    )


def open_stage(
    name: str,
    description: StageDescription,
    components: Mapping[str, object],
    stage_types: StageTypes,
    context: StageContext,
) -> BaseStage[Any, Any] | CallStage:
    """Build one stage: its stage type with its settings, or a plain call.

    Args:
        name: The stage's name, which its type defaults to.
        description: The stage's type, ``call``, lists and own settings.
        components: What ``call`` names, by name.
        stage_types: Where the stage's type is found.
        context: What a stage of a stage type is given; a plain call needs none.

    Returns:
        The stage: of its stage type, or a plain call when it is of none.

    Raises:
        WorkflowGraphError: Its ``type`` names no stage type, its settings or
            lists do not fit its stage type, its stage type refuses the
            component, or a plain call's keys and lists do not bind to its
            component; the message names the stage.

    Example:
        >>> open_stage("retrieve", StageDescription.model_validate(
        ...     {"call": "vector_store", "retrieval_mode": "dense", "top_k": 20,
        ...      "reads": ["query_embedding"], "writes": ["candidates"]}),
        ...     components, STAGE_TYPES, StageContext())
        <RetrieveStage ...>
        >>> open_stage("embed", StageDescription.model_validate(
        ...     {"call": "embedder", "reads": ["standalone_query"],
        ...      "writes": ["query_embedding"]}), components, STAGE_TYPES,
        ...     StageContext())
        <CallStage ...>
    """
    component = components[description.call]
    try:
        stage_type = stage_type_of(name, description, stage_types)
        if stage_type is None:
            return open_call(description, component)
        settings = stage_type.settings_type.model_validate(description.stage_settings)
        return stage_type(component, settings, stage_fields(description), context)
    except ValueError as exc:
        # A refusal, so qp-serve reports it as one line, not as a crash.
        raise WorkflowGraphError(f"stage {name!r}: {exc}") from exc


def open_call(description: StageDescription, component: object) -> CallStage:
    """Build a plain call: its component, bound by its keys and its lists.

    Args:
        description: The stage, which states ``reads`` and ``writes``.
        component: What its ``call`` names, a component with ``ainvoke``.

    Returns:
        The plain call.

    Raises:
        ValueError: The component is no ``BaseComponent``, or the stage's keys
            and lists do not bind to its request and answer.

    Example:
        >>> open_call(StageDescription.model_validate({"call": "embedder",
        ...     "reads": ["standalone_query"], "writes": ["query_embedding"]}),
        ...     embedder)
        <CallStage ...>
    """
    if not isinstance(component, BaseComponent):
        raise ValueError(
            f"a plain call calls {description.call!r}, a "
            f"{type(component).__name__} (expected a component of a component "
            "type, which answers ainvoke)"
        )
    fields = stage_fields(description)
    binding = bind_call(type(component), description.stage_settings, fields)
    return CallStage(component, binding)


def timed(name: str, stage: BaseStage[Any, Any] | CallStage) -> StageRun:
    """Wrap a stage so its update carries its measures under its name.

    The update gains ``{"timings_ms": {name: ms}}``, and the tokens the stage
    hands unnamed, ``{"tokens": counts}``, become ``{"tokens": {name: counts}}``.

    Args:
        name: The stage's name, the key its measures are recorded under.
        stage: The stage to time.

    Returns:
        What LangGraph runs for the stage.

    Example:
        >>> builder.add_node("rerank", timed("rerank", stage))
    """

    async def run(state: WorkflowState) -> WorkflowState:
        started = time.perf_counter()
        update = await stage.run(state)
        elapsed_ms = (time.perf_counter() - started) * 1000
        measured: WorkflowState = {TIMINGS_FIELD: {name: elapsed_ms}}
        if TOKENS_FIELD in update:
            measured[TOKENS_FIELD] = {name: update[TOKENS_FIELD]}
        return {**update, **measured}

    return run
