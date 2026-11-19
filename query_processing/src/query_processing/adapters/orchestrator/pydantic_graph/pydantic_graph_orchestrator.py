"""``PydanticGraphOrchestrator``: pydantic-graph runs the opened graph.

The one module that imports pydantic-graph (its builder API, ``GraphBuilder``).
Each opened stage becomes a step of its name, which hands the stage a copy of
the workflow state and keeps the state the opened graph's ``merged`` gives: the
library has no reducers. Each conditional edge becomes a step named
``<source>:<edge condition>`` that asks the edge condition, then a decision
with one branch for each row.

Its class refuses what the library does not run as LangGraph does (ADR 0010,
seen on 2.70.0): more than one way out of a stage at once, since the library
ends a run when the first branch reaches ``END`` and cancels the others; and a
loop, since nothing stops one that never ends.

Example:
    >>> orchestrator = PydanticGraphOrchestrator(opened)
    >>> final = await orchestrator.ainvoke(initial_state("q?", "query"))
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any

from pydantic_graph import Graph, GraphBuilder, StepContext, TypeExpression
from pydantic_graph.step import StepFunction

from ....domain.graph_description import END, START, GraphDescription
from ....domain.opened_graph import (
    EdgeAsk,
    OpenedConditionalEdge,
    OpenedGraph,
    StageRun,
)
from ..base_orchestrator import ORCHESTRATORS, BaseOrchestrator

# What every refusal offers instead: the reference, which runs every graph.
_INSTEAD = "or the orchestrator 'langgraph'"


@dataclass
class _RunState:
    """One run's state: the workflow state of one request, replaced at each step."""

    values: dict[str, Any]


# A step's context: the run's state, no dependencies, any input.
_Context = StepContext[_RunState, None, object]
# A step of this graph: asked with its context, it returns any output.
_Step = StepFunction[_RunState, None, object, object]


@ORCHESTRATORS.register("pydantic_graph")
class PydanticGraphOrchestrator(BaseOrchestrator):
    """pydantic-graph's graph of the opened stages and edges.

    Args:
        opened: The stages, the edges, the conditional edges, the field types
            and the reducers.

    Example:
        >>> "route:label" in PydanticGraphOrchestrator(opened).graph.nodes
        True
    """

    @classmethod
    def check_graph_runs(cls, graph: GraphDescription) -> None:
        """Refuse more than one way out of a stage at once, and a loop.

        Args:
            graph: The workflow file's graph, accepted by the toolkit's checks.

        Raises:
            ValueError: Naming the stages, what pydantic-graph would do, and
                what was expected.

        Example:
            >>> PydanticGraphOrchestrator.check_graph_runs(two_edges_out_of_start)
            ValueError: 'START' has more than one way out at once; ...
        """
        _check_one_way_out(graph)
        _check_no_loop(graph)

    def __init__(self, opened: OpenedGraph) -> None:
        builder = GraphBuilder(
            name="workflow", state_type=_RunState, output_type=object
        )
        nodes: dict[str, Any] = {START: builder.start_node, END: builder.end_node}
        for name, run in opened.stages.items():
            nodes[name] = builder.step(_stage_step(run, opened), node_id=name)
        for source, target in opened.edges:
            builder.add_edge(nodes[source], nodes[target])
        for conditional in opened.conditional_edges:
            _add_conditional_edge(builder, nodes, conditional)
        self.graph: Graph[_RunState, None, None, object] = builder.build()

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run one request through pydantic-graph's graph.

        Args:
            state: The state the request starts with; not changed.

        Returns:
            The state once the run reached ``END``.

        Example:
            >>> (await orchestrator.ainvoke(initial_state("q?", "query")))["answer"]
            'Q?!'
        """
        run = _RunState(dict(state))
        await self.graph.run(state=run, infer_name=False)
        return run.values


def _check_one_way_out(graph: GraphDescription) -> None:
    ways_out = Counter(source for source, _ in graph.edge_pairs())
    ways_out.update(edge.source for edge in graph.conditional_edges)
    several = [source for source, count in ways_out.items() if count > 1]
    if not several:
        return
    have = "has" if len(several) == 1 else "each have"
    raise ValueError(
        f"{_quoted(several)} {have} more than one way out at once; pydantic-graph "
        "ends a run when the first reaches END and cancels the others (expected "
        f"one plain edge or one conditional edge out of each stage, {_INSTEAD})"
    )


def _check_no_loop(graph: GraphDescription) -> None:
    sorter: TopologicalSorter[str] = TopologicalSorter()
    for source, target in graph.links():
        sorter.add(target, source)
    try:
        sorter.prepare()
    except CycleError as exc:
        loop = " -> ".join(repr(stage) for stage in exc.args[1])
        raise ValueError(
            f"the stages {loop} form a loop; pydantic-graph has no limit that "
            f"stops one (expected a graph without a loop, {_INSTEAD})"
        ) from exc


def _add_conditional_edge(
    builder: GraphBuilder[_RunState, None, None, object],
    nodes: dict[str, Any],
    conditional: OpenedConditionalEdge,
) -> None:
    edge = conditional.edge
    # Unique in the graph: the toolkit uses an edge condition once per source.
    step_name = f"{edge.source}:{edge.path}"
    # Any, as the nodes are: ty 0.0.100 matches neither a step's output against
    # the library's Source[object] nor a Decision[..., Never] against a
    # destination, which its own examples pass.
    asked: Any = builder.step(_ask_step(conditional.ask), node_id=step_name)
    decision: Any = builder.decision()
    for answer, target in edge.path_map.items():
        row = builder.match(TypeExpression[object], matches=_is(answer))
        decision = decision.branch(row.to(nodes[target]))
    builder.add_edge(nodes[edge.source], asked)
    builder.add_edge(asked, decision)


def _stage_step(run: StageRun, opened: OpenedGraph) -> _Step:
    # A copy, as LangGraph hands a node the state: a stage that changes what it
    # is handed changes nothing of the run's state.
    async def step(ctx: _Context) -> None:
        update = await run(dict(ctx.state.values))
        ctx.state.values = opened.merged(ctx.state.values, update)

    return step


def _ask_step(ask: EdgeAsk) -> _Step:
    async def step(ctx: _Context) -> object:
        return ask(ctx.state.values)

    return step


def _is(answer: object) -> Callable[[object], bool]:
    # The row's own type too: 1 == True in Python, and 1 must not take True's row.
    return lambda got: type(got) is type(answer) and got == answer


def _quoted(names: list[str]) -> str:
    return ", ".join(repr(name) for name in names)
