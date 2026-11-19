"""``PydanticGraphOrchestrator``: pydantic-graph runs the opened graph.

The one module that imports pydantic-graph (its builder API, ``GraphBuilder``).
Each opened stage becomes a step of its name, which hands the stage a copy of
the workflow state and keeps the state the opened graph's ``merged`` gives: the
library has no reducers. Each conditional edge becomes a step named
``<source>:<edge condition>`` that asks the edge condition, then a decision
named ``<source>:<edge condition>:decision`` with one branch for each row:
every node the adapter adds has a name of its own, never the library's. One
function, ``_linked``, links the library's graph, for the build and the check.

Its class refuses what the library does not run as LangGraph does (ADR 0010,
seen on 2.70.0): a stage named as a node the adapter adds; more than one way
out of a stage at once, since the library ends a run when the first branch
reaches ``END`` and cancels the others; a loop, since nothing stops one that
never ends; and a graph the library refuses to build (a stage with no way
out, nothing that leads to ``END``, a stage no edge reaches), in its words.

Example:
    >>> orchestrator = PydanticGraphOrchestrator(opened)
    >>> final = await orchestrator.ainvoke(initial_state("q?", "query"))
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any, cast

from pydantic_graph import Graph, GraphBuilder, StepContext
from pydantic_graph.exceptions import GraphBuildingError, GraphValidationError
from pydantic_graph.step import StepFunction

from ....domain.graph_description import (
    END,
    START,
    ConditionalEdge,
    GraphDescription,
)
from ....domain.opened_graph import (
    EdgeAsk,
    OpenedConditionalEdge,
    OpenedGraph,
    StageRun,
)
from ..base_orchestrator import ORCHESTRATORS, BaseOrchestrator


@dataclass
class _RunState:
    """One run's state: the workflow state of one request, replaced at each step."""

    values: dict[str, Any]


# The library's classes are written here without their generic parameters: it
# declares its type variables as typing_extensions.TypeVar(..., infer_variance=
# True) (pydantic_graph/step.py, util.py), which mypy 2.4 does not take on a
# plain TypeVar, so for mypy no class of it is generic and StepContext[...]
# is an error. The run's state is typed through _run_state instead.


@ORCHESTRATORS.register("pydantic_graph")
class PydanticGraphOrchestrator(BaseOrchestrator):
    """pydantic-graph's graph of the opened stages and edges.

    Args:
        opened: The stages, the edges, the conditional edges, the field types
            and the reducers.

    Example:
        >>> "route:label:decision" in PydanticGraphOrchestrator(opened).graph.nodes
        True
    """

    @classmethod
    def check_graph_runs(cls, graph: GraphDescription) -> None:
        """Refuse a graph pydantic-graph would not build, or not run as LangGraph does.

        A stage named as a node of the adapter's own, more than one way out
        of a stage at once, a loop; then what the library refuses to build,
        asked of it on the graph linked as the build links it, stand-in steps
        in place of the stages.

        Args:
            graph: The workflow file's graph, accepted by the toolkit's checks.

        Raises:
            ValueError: Naming the stages, what pydantic-graph would do, and
                what was expected.

        Example:
            >>> PydanticGraphOrchestrator.check_graph_runs(two_edges_out_of_start)
            ValueError: 'START' has more than one way out at once; ...
        """
        _check_no_stage_named_as_a_node_of_its_own(graph)
        _check_one_way_out(graph)
        _check_no_loop(graph)
        _check_the_library_builds(graph)

    def __init__(self, opened: OpenedGraph) -> None:
        steps = {name: _stage_step(run, opened) for name, run in opened.stages.items()}
        self.graph: Graph = _linked(steps, opened.edges, opened.conditional_edges)

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


def _check_no_stage_named_as_a_node_of_its_own(graph: GraphDescription) -> None:
    # The library refuses a name used twice too, but in words that show its
    # objects' reprs; and a stage of such a name would take the node's place.
    for edge in graph.conditional_edges:
        own = (_asked_name(edge), _decision_name(edge))
        taken = [name for name in own if name in graph.stages]
        if taken:
            raise ValueError(
                f"stage {taken[0]!r} has the name of a node this orchestrator adds "
                f"for the conditional edge out of {edge.source!r} (expected a "
                f"stage name other than {own[0]!r} and {own[1]!r})"
            )


def _asked_name(edge: ConditionalEdge) -> str:
    # Unique in the graph: the toolkit uses an edge condition once per source.
    return f"{edge.source}:{edge.path}"


def _decision_name(edge: ConditionalEdge) -> str:
    # Named here, not by the library: it would name it decision, decision_2,
    # ..., in the stages' own namespace, and a stage of that name replaces it.
    return f"{_asked_name(edge)}:decision"


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
        "one plain edge or one conditional edge out of each stage)"
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
            "stops one (expected a graph without a loop)"
        ) from exc


def _check_the_library_builds(graph: GraphDescription) -> None:
    # The graph linked as the build links it, each stage a step that is never
    # run: what the library refuses to build is refused before anything opens.
    steps = dict.fromkeys(graph.stages, _never_run)
    asked = [OpenedConditionalEdge(e, _never_asked) for e in graph.conditional_edges]
    try:
        _linked(steps, graph.edge_pairs(), asked)
    except (GraphBuildingError, GraphValidationError) as exc:
        said = " ".join(str(exc).split())  # one line, as qp-serve prints it
        raise ValueError(
            f"pydantic-graph does not build this graph: {said} (expected every "
            "stage on a path from START to END)"
        ) from exc


def _linked(
    steps: Mapping[str, StepFunction],
    edges: Sequence[tuple[str, str]],
    conditional_edges: Sequence[OpenedConditionalEdge],
) -> Graph:
    # The one place the library's graph is linked, for the build and the check.
    builder = GraphBuilder(name="workflow", state_type=_RunState, output_type=object)
    nodes: dict[str, Any] = {START: builder.start_node, END: builder.end_node}
    for name, step in steps.items():
        nodes[name] = builder.step(step, node_id=name)
    for source, target in edges:
        builder.add_edge(nodes[source], nodes[target])
    for conditional in conditional_edges:
        _add_conditional_edge(builder, nodes, conditional)
    return builder.build()


async def _never_run(ctx: StepContext) -> None:
    # A stage's stand-in in the check, which links the graph and runs nothing.
    del ctx  # the library hands a step its context; the stand-in reads none
    raise AssertionError("a stand-in step of the check ran")


def _never_asked(state: dict[str, Any]) -> bool:
    # An edge condition's stand-in in the check, which asks nothing.
    del state  # an edge condition is asked of the state; the stand-in reads none
    raise AssertionError("a stand-in edge condition of the check was asked")


def _add_conditional_edge(
    builder: GraphBuilder,
    nodes: dict[str, Any],
    conditional: OpenedConditionalEdge,
) -> None:
    edge = conditional.edge
    # Any, as the nodes are: ty 0.0.100 matches neither a step's output against
    # the library's Source[object] nor a Decision[..., Never] against a
    # destination, which its own examples pass.
    asked: Any = builder.step(_ask_step(conditional.ask), node_id=_asked_name(edge))
    decision: Any = builder.decision(node_id=_decision_name(edge))
    for answer, target in edge.path_map.items():
        row = builder.match(object, matches=_is(answer))
        decision = decision.branch(row.to(nodes[target]))
    builder.add_edge(nodes[edge.source], asked)
    builder.add_edge(asked, decision)


def _run_state(ctx: StepContext) -> _RunState:
    # The state handed to graph.run in ainvoke, untyped for mypy (see above).
    return cast(_RunState, ctx.state)


def _stage_step(run: StageRun, opened: OpenedGraph) -> StepFunction:
    # A copy, as LangGraph hands a node the state: a stage that changes what it
    # is handed changes nothing of the run's state.
    async def step(ctx: StepContext) -> None:
        state = _run_state(ctx)
        update = await run(dict(state.values))
        state.values = opened.merged(state.values, update)

    return step


def _ask_step(ask: EdgeAsk) -> StepFunction:
    async def step(ctx: StepContext) -> object:
        return ask(_run_state(ctx).values)

    return step


def _is(answer: object) -> Callable[[object], bool]:
    # Equal, and a bool only for a bool: 1 == True in Python, and 1 must not take
    # True's row; a label of a str subclass (a StrEnum) takes its str row.
    is_bool = isinstance(answer, bool)
    return lambda got: isinstance(got, bool) == is_bool and got == answer


def _quoted(names: list[str]) -> str:
    return ", ".join(repr(name) for name in names)
