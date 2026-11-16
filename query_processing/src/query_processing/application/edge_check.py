"""The start-up checks of a graph's edges, plain and conditional (spec 3.4).

A plain edge goes from ``START`` or a stage to a stage or ``END``. A conditional
edge starts where a plain edge may, names a registered edge condition, reads one
field for each parameter of it, and has one row of ``path_map`` for each answer
it can give, each leading where a plain edge may: a yes-or-no edge condition a
``true`` and a ``false`` row, and a label edge (ADR 0007) one row for each label
the writer of the one field it reads declares. An edge condition is asked once
per source. Whether the fields an edge reads are set when it is asked is the
field rule of ``graph_check``.

Example:
    >>> check_edges(graph, edge_conditions, labels={"route": ("self_contained",)})
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping

from ..domain.graph_description import END, START, ConditionalEdge, GraphDescription
from ..domain.registry import Registry
from .edge_conditions.edge_condition import EdgeCondition, answers_with_label
from .stages.base_stage import FieldTypes, field_count
from .state_fields import WorkflowGraphError

# The kinds of parameter a field's value can be handed to, by position.
_POSITIONAL = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def check_edges(
    graph: GraphDescription,
    edge_conditions: Registry[EdgeCondition],
    labels: Mapping[str, tuple[str, ...]],
) -> None:
    """Refuse an edge that cannot run, and a graph that nothing enters.

    Args:
        graph: The stages, the chains of plain edges and the conditional edges.
        edge_conditions: Where a conditional edge's ``path`` is found.
        labels: The labels each field can take, as ``field_labels`` gives.

    Raises:
        WorkflowGraphError: An edge starts or ends at no stage; a conditional
            edge names an unknown edge condition, reads another number of
            fields than it takes, lacks a row for one of its answers or has
            one for an answer it cannot give, or repeats an edge condition on
            one source; nothing leaves START.

    Example:
        >>> check_edges(settings.graph, EDGE_CONDITIONS, {})
    """
    for source, target in graph.edge_pairs():
        edge = f"edge {source!r} -> {target!r}"
        _check_source(graph, source, edge)
        _check_target(graph, target, edge)
    _check_conditional_edges(graph, edge_conditions, labels)
    _check_entry(graph)


def field_labels(fields: Mapping[str, FieldTypes]) -> dict[str, tuple[str, ...]]:
    """Return the labels each field can take, as the stages that write it declare.

    Args:
        fields: What each stage reads and writes, by stage name.

    Returns:
        The labels of each field a stage declares some for.

    Raises:
        WorkflowGraphError: Two stages declare other labels for one field.

    Example:
        >>> field_labels({"route": FieldTypes({}, {"route": str},
        ...                                   {"route": ("self_contained",)})})
        {'route': ('self_contained',)}
    """
    declared = [
        (stage, field, given)
        for stage, types in fields.items()
        for field, given in types.labels.items()
    ]
    first: dict[str, tuple[str, tuple[str, ...]]] = {}
    for stage, field, given in declared:
        writer, labels = first.setdefault(field, (stage, given))
        if set(labels) != set(given):
            raise WorkflowGraphError(
                f"{field!r} takes the labels {quoted(labels)} by {writer!r} and "
                f"{quoted(given)} by {stage!r} (expected every writer of a field "
                "to declare the same labels)"
            )
    return {field: labels for field, (_, labels) in first.items()}


def links(graph: GraphDescription) -> list[tuple[str, str]]:
    """Return every way from one stage to the next: each plain edge, each answer's row.

    Args:
        graph: The chains of plain edges and the conditional edges.

    Returns:
        ``(source, target)`` for each plain edge, then for each row of each
        conditional edge's ``path_map``.

    Example:
        >>> links(graph)
        [('START', 'embed'), ..., ('rerank', 'generate'), ('rerank', 'END')]
    """
    answers = [
        (edge.source, target)
        for edge in graph.conditional_edges
        for target in edge.path_map.values()
    ]
    return [*graph.edge_pairs(), *answers]


def edge_name(edge: ConditionalEdge) -> str:
    """Return a conditional edge as a refusal names it.

    Args:
        edge: The conditional edge.

    Returns:
        Its source and its edge condition, in words.

    Example:
        >>> edge_name(edge)
        "conditional edge from 'rerank' on 'is_query_mode'"
    """
    return f"conditional edge from {edge.source!r} on {edge.path!r}"


def quoted(names: Iterable[str]) -> str:
    """Return names as a refusal lists them: quoted, sorted, comma-separated.

    Args:
        names: The names to list.

    Returns:
        Each name's ``repr``, sorted, joined by commas.

    Example:
        >>> quoted(["b", "a"])
        "'a', 'b'"
    """
    return ", ".join(repr(name) for name in sorted(names))


def _check_source(graph: GraphDescription, source: str, edge: str) -> None:
    # `edge` names the edge in the refusal.
    if source != START and source not in graph.stages:
        raise WorkflowGraphError(
            f"{edge} starts at {source!r}, which is no stage "
            f"(expected START or one of the stages: {quoted(graph.stages)})"
        )


def _check_target(graph: GraphDescription, target: str, edge: str) -> None:
    if target != END and target not in graph.stages:
        raise WorkflowGraphError(
            f"{edge} leads to {target!r}, which is no stage "
            f"(expected END or one of the stages: {quoted(graph.stages)})"
        )


def _check_conditional_edges(
    graph: GraphDescription,
    edge_conditions: Registry[EdgeCondition],
    labels: Mapping[str, tuple[str, ...]],
) -> None:
    asked: set[tuple[str, str]] = set()
    for edge in graph.conditional_edges:
        _check_conditional_edge(graph, edge, edge_conditions, labels)
        if (edge.source, edge.path) in asked:
            # LangGraph would refuse it too, naming the branch, not the file.
            raise WorkflowGraphError(
                f"edge condition {edge.path!r} is used twice on {edge.source!r} "
                "(expected each edge condition once per source: LangGraph names a "
                "branch after its edge condition, and one conditional edge "
                "already says where each of its answers leads)"
            )
        asked.add((edge.source, edge.path))


def _check_conditional_edge(
    graph: GraphDescription,
    edge: ConditionalEdge,
    edge_conditions: Registry[EdgeCondition],
    labels: Mapping[str, tuple[str, ...]],
) -> None:
    where = edge_name(edge)
    _check_source(graph, edge.source, where)
    try:
        condition = edge_conditions.get(edge.path)
    except ValueError as exc:  # the registry names the known edge conditions
        raise WorkflowGraphError(f"{where}: {exc}") from exc
    _check_reads_fill(edge, condition)
    if answers_with_label(condition):
        _check_label_rows(edge, labels)
    else:
        _check_yes_no_rows(edge)
    for answer, target in edge.path_map.items():
        _check_target(graph, target, f"the {_said(answer)} answer of the {where}")


def _check_reads_fill(edge: ConditionalEdge, condition: EdgeCondition) -> None:
    # The values of reads are handed to the edge condition by position.
    parameters = inspect.signature(condition).parameters.values()
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters):
        return
    takes = [p.name for p in parameters if p.kind in _POSITIONAL]
    if len(takes) != len(edge.reads):
        raise WorkflowGraphError(
            f"{edge_name(edge)} reads {field_count(len(edge.reads))} "
            f"{edge.reads}, but {edge.path!r} takes ({', '.join(takes)}) "
            f"(expected reads to list {field_count(len(takes))}, one for each)"
        )


def _check_yes_no_rows(edge: ConditionalEdge) -> None:
    for answer in (True, False):
        if answer not in edge.path_map:
            raise WorkflowGraphError(
                f"{edge_name(edge)} has no {_said(answer)} target (expected a "
                "path_map with a true and a false target, each a stage or END)"
            )
    others = [str(key) for key in edge.path_map if not isinstance(key, bool)]
    if others:
        raise WorkflowGraphError(
            f"{edge_name(edge)} has rows for {quoted(others)}, which "
            f"{edge.path!r} never answers (expected a path_map with a true and a "
            "false target only, since it answers yes or no)"
        )


def _check_label_rows(
    edge: ConditionalEdge, labels: Mapping[str, tuple[str, ...]]
) -> None:
    where = edge_name(edge)
    if len(edge.reads) != 1 or edge.reads[0] not in labels:
        raise WorkflowGraphError(
            f"{where} reads {edge.reads}, which has no declared labels (expected "
            "one field whose writer declares its labels, such as a router's route)"
        )
    if any(isinstance(answer, bool) for answer in edge.path_map):
        # The file's true and false are yes and no: label answers it never gives.
        raise WorkflowGraphError(
            f"{where}: path_map has a true or a false row, which a label never "
            "takes (expected labels other than true and false, which the "
            "workflow file reads as yes and no)"
        )
    expected = labels[edge.reads[0]]
    listed = [str(answer) for answer in edge.path_map]
    if set(listed) != set(expected):
        raise WorkflowGraphError(
            f"{where}: path_map lists {quoted(listed)} (expected one row for "
            f"each label {edge.reads[0]!r} takes, and no other: {quoted(expected)})"
        )


def _said(answer: bool | str) -> str:
    # An answer as the workflow file writes it.
    return str(answer).lower() if isinstance(answer, bool) else answer


def _check_entry(graph: GraphDescription) -> None:
    if all(source != START for source, _ in links(graph)):
        raise WorkflowGraphError(
            "the graph has no edge from START (expected a chain of plain edges "
            "that starts with START, or a conditional edge from START)"
        )
