"""``LangGraphOrchestrator``: LangGraph runs the opened graph, the default orchestrator.

The one module that imports LangGraph. Each opened stage becomes a node of its
name; each plain edge one LangGraph edge, the reserved ``START`` and ``END``
becoming LangGraph's own; each conditional edge asks its edge condition through
a function that keeps the edge condition's name, since LangGraph names the
branch after the function and Domino tracing needs that name. The state is a
``TypedDict`` of the opened graph's field types, built at run time, each field
that adds up annotated with its reducer.

Example:
    >>> orchestrator = LangGraphOrchestrator(opened)
    >>> final = await orchestrator.ainvoke(initial_state("q?", "query"))
"""

from __future__ import annotations

from collections.abc import Awaitable, Hashable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypedDict, cast

from langgraph.graph import END as LANGGRAPH_END
from langgraph.graph import START as LANGGRAPH_START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from ....domain.graph_description import END, START, ConditionalEdge, GraphDescription
from ....domain.opened_graph import OpenedGraph, Reducer
from ..base_orchestrator import ORCHESTRATORS, BaseOrchestrator

# The description's reserved names, as LangGraph spells them.
_LANGGRAPH_NAMES = {START: LANGGRAPH_START, END: LANGGRAPH_END}


class LangGraphNode(Protocol):
    """What LangGraph runs for one stage (LangGraph calls it a node).

    A Protocol rather than ``Callable[[WorkflowState], ...]`` because LangGraph
    1.x types a node by a parameter named ``state``; a ``Callable`` alias erases
    the name and mypy then rejects ``StateGraph.add_node``. Its state is
    ``Any``: the state type is built at run time, and LangGraph bounds a node's
    input by a TypedDict-like type that a plain ``dict`` is not.
    """

    def __call__(self, state: Any) -> Awaitable[dict[str, Any]]:
        """Run the stage.

        Args:
            state: The workflow state so far.

        Returns:
            An awaitable of the stage's update.
        """


@ORCHESTRATORS.register("langgraph")
class LangGraphOrchestrator(BaseOrchestrator):
    """LangGraph's compiled graph of the opened stages and edges.

    Args:
        opened: The stages, the edges, the conditional edges, the field types
            and the reducers.

    Example:
        >>> LangGraphOrchestrator(opened).graph.get_graph().edges
        [Edge(source='__start__', target='draft', ...), ...]
    """

    @classmethod
    def check_graph_runs(cls, graph: GraphDescription) -> None:
        """Refuse a stage name LangGraph does not take for a node, in its words.

        LangGraph itself is asked: each name is added to a throwaway graph,
        with a stand-in node, so what it refuses is its own rule, not a list
        written here. It is the reference the other orchestrators are held to,
        the library the workflow file's names come from (ADR 0010).

        Args:
            graph: The workflow file's graph, accepted by the toolkit's checks.

        Raises:
            ValueError: Naming the stage and LangGraph's reason.

        Example:
            >>> LangGraphOrchestrator.check_graph_runs(a_stage_named("re:rank"))
            ValueError: stage 're:rank' is no name LangGraph takes for a node: ...
        """
        for name in graph.stages:
            _check_langgraph_takes(name)

    def __init__(self, opened: OpenedGraph) -> None:
        builder = StateGraph(state_type(opened.field_types, opened.reducers))
        for name, run in opened.stages.items():
            builder.add_node(name, cast(LangGraphNode, run))
        for source, target in opened.edges:
            builder.add_edge(langgraph_name(source), langgraph_name(target))
        for conditional in opened.conditional_edges:
            source = langgraph_name(conditional.edge.source)
            builder.add_conditional_edges(
                source, conditional.ask, path_map(conditional.edge)
            )
        # Any, not the state type: ty 0.0.100 does not match a TypedDict against
        # LangGraph's TypedDictLike bound (its __optional_keys__); mypy accepts it.
        self.graph: CompiledStateGraph[Any, Any, Any, Any] = builder.compile()

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run one request through LangGraph's compiled graph.

        Args:
            state: The state the request starts with.

        Returns:
            The state once the run reached ``END``.

        Example:
            >>> (await orchestrator.ainvoke(initial_state("q?", "query")))["answer"]
            'Q?!'
        """
        return await self.graph.ainvoke(state)


# A dataclass, not a TypedDict: ty 0.0.100 does not match a TypedDict against
# LangGraph's TypedDictLike bound (see __init__).
@dataclass
class _NoFields:
    """The throwaway graph's state: no field, so only a node's name is asked."""


def _check_langgraph_takes(name: str) -> None:
    try:
        StateGraph(_NoFields).add_node(name, _never_run)
    except ValueError as exc:
        raise ValueError(
            f"stage {name!r} is no name LangGraph takes for a node: {exc} "
            "(expected a stage name LangGraph takes)"
        ) from exc


async def _never_run(state: Any) -> dict[str, Any]:
    # The stand-in node of the throwaway graph, which is never compiled.
    del state  # LangGraph hands a node the state; the stand-in reads none
    raise AssertionError("the stand-in node of the check ran")


def state_type(
    field_types: Mapping[str, Any], reducers: Mapping[str, Reducer]
) -> type[Any]:
    """Return the LangGraph state of these fields, those that add up with their reducer.

    Args:
        field_types: Every field of the workflow state, with its type.
        reducers: The fields that add up, each with its reducer.

    Returns:
        A ``TypedDict`` in which no field is required: a field is set when its
        writer has run. A field of ``reducers`` is annotated with its reducer.

    Example:
        >>> get_type_hints(state_type({"draft": str}, {}))["draft"]
        <class 'str'>
    """
    fields = {
        field: Annotated[kind, reducers[field]] if field in reducers else kind
        for field, kind in field_types.items()
    }
    # The functional form, since the fields are known only at run time; mypy
    # types only a literal dict there.
    make_typed_dict = cast(Any, TypedDict)
    return cast(type[Any], make_typed_dict("WorkflowState", fields, total=False))


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
        ...          reads=["mode"], path_map={True: "generate", False: "END"}))
        {True: 'generate', False: '__end__'}
    """
    return {answer: langgraph_name(to) for answer, to in edge.path_map.items()}
