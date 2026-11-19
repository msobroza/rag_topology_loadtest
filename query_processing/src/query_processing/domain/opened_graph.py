"""The opened graph: a graph description with each stage now what it runs.

``build_workflow`` opens each stage, times it and asks each conditional edge's
edge condition of the state, with no library; an orchestrator is given the
result and links it with its own library (ADR 0010). The opened graph keeps
the description's three keys: the stages, by name, in the file's order; the
plain edges as pairs; each conditional edge as the description's own, with
what it asks of the state. It also holds the type of every field of the
workflow state, and the reducer of each field that adds up.

It is in the domain because the orchestrator's port, which imports the domain
only, is built from it.

Example:
    >>> opened = OpenedGraph(stages={"draft": run_draft},
    ...     edges=[("START", "draft"), ("draft", "END")], conditional_edges=[],
    ...     field_types={"draft": str}, reducers={})
    >>> list(opened.stages)
    ['draft']
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .graph_description import ConditionalEdge

# One stage, opened and timed: the workflow state in, the stage's update out.
StageRun = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
# A conditional edge's edge condition asked of the state: the answer of a row.
EdgeAsk = Callable[[dict[str, Any]], bool | str]
# How a field that adds up takes an update: its value so far and the update's.
Reducer = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class OpenedConditionalEdge:
    """A conditional edge of the description, with what it asks of the state.

    Args:
        edge: The description's conditional edge: its source, its edge
            condition (``path``) and its rows (``path_map``).
        ask: Its edge condition asked of the state; an answer with no row is
            refused.

    Example:
        >>> OpenedConditionalEdge(edge, asked_of_state(is_query_mode, edge)).edge.path
        'is_query_mode'
    """

    edge: ConditionalEdge
    ask: EdgeAsk


@dataclass(frozen=True)
class OpenedGraph:
    """What an orchestrator is built from: the graph, each stage opened.

    Args:
        stages: Each stage, timed, by name, in the file's order.
        edges: The plain edges, the description's ``edge_pairs()``.
        conditional_edges: Each conditional edge, with what it asks.
        field_types: Every field of the workflow state, with its type.
        reducers: The fields that add up, each with its reducer.

    Example:
        >>> opened.edges
        [('START', 'draft'), ('draft', 'END')]
    """

    stages: Mapping[str, StageRun]
    edges: Sequence[tuple[str, str]]
    conditional_edges: Sequence[OpenedConditionalEdge]
    field_types: Mapping[str, Any]
    reducers: Mapping[str, Reducer]
