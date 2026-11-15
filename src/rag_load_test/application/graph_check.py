"""``check_graph``: a workflow graph that cannot run, refused before it is built.

The checks know no stage type, no component and no edge condition. They read
what each stage type declares (the port of the component it calls, its
settings, and the workflow state fields it reads and writes), the declared
components with their types, and the registered edge conditions. All of it is
in the workflow file and the settings, so ``rag-serve`` runs them before any
component opens, and a refusal names what is wrong and what was expected. What
only an opened component shows, a store that ranks no text under a lexical
retrieval, is refused when the graph is built, as a ``WorkflowGraphError`` too.

The field rule: a stage may read a workflow state field only if the request
provides it or a stage writes it on every path from ``START`` to the stage. A
dense retrieval with no embed stage before it is refused by that rule, not by a
rule of its own (ADR 0004).

Example:
    >>> check_graph(settings.graph, declared_components())  # today's: nothing raised
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from pydantic import ValidationError

from ..domain.graph_description import (
    END,
    START,
    ConditionalEdge,
    GraphDescription,
    StageDescription,
)
from ..domain.passages import TOTAL_TIMING_KEY
from ..domain.registry import Registry
from .component_types import COMPONENT_TYPES
from .edge_conditions.edge_condition import EDGE_CONDITIONS, EdgeCondition
from .stages.base_stage import STAGE_TYPES, BaseStage, StageSettings, StageTypes
from .workflow_state import REQUEST_FIELDS

# A stage named like one of these could not be told from it.
_RESERVED_STAGE_NAMES = {
    START: "the start of the graph in its edges",
    END: "the end of the graph in its edges",
    TOTAL_TIMING_KEY: (
        "the key of the total time in timings_ms, where its timing would collide "
        "with the total"
    ),
}


class WorkflowGraphError(ValueError):
    """A workflow graph that cannot run: the message says what and what was expected."""


class DeclaredFields(NamedTuple):
    """The workflow state fields one stage reads and writes, as its stage type says."""

    reads: frozenset[str]
    writes: frozenset[str]


def check_graph(
    graph: GraphDescription,
    components: Mapping[str, str],
    *,
    component_types: Mapping[str, type[Any]] = COMPONENT_TYPES,
    stage_types: StageTypes = STAGE_TYPES,
    edge_conditions: Registry[EdgeCondition] = EDGE_CONDITIONS,
) -> None:
    """Refuse a graph that cannot run, naming the first mistake found.

    Args:
        graph: The stages, the chains of plain edges and the conditional edges.
        components: Each declared component's type, by the component's name.
        component_types: The port of each component type, by the type's name.
        stage_types: Where a stage's type is found.
        edge_conditions: Where a conditional edge's ``path`` is found.

    Raises:
        WorkflowGraphError: A stage has a reserved name, an unknown stage type,
            settings its stage type refuses, or calls a missing component or one
            of the wrong type; an edge names no stage; a conditional edge names
            an unknown edge condition, lacks its true or false target, or repeats
            an edge condition on one source; nothing leaves START; or a stage
            reads a field that is not sure to be set when it runs.

    Example:
        >>> check_graph(settings.graph, declared_components())
    """
    _check_stage_names(graph)
    fields = {
        name: _declared_fields(name, stage, components, component_types, stage_types)
        for name, stage in graph.stages.items()
    }
    _check_edges(graph)
    _check_conditional_edges(graph, edge_conditions)
    _check_entry(graph)
    _check_fields_read(graph, fields)


def fields_set_before(
    graph: GraphDescription, writes: Mapping[str, frozenset[str]]
) -> dict[str, frozenset[str]]:
    """Return, for each stage, the workflow state fields sure to be set when it runs.

    A field is sure to be set when the request provides it, or when a stage
    writes it on every path from ``START`` to the stage. A stage no path reaches
    never runs, so every field counts as set for it.

    Args:
        graph: The stages and edges; every edge names a stage, START or END.
        writes: The fields each stage writes, by the stage's name.

    Returns:
        The fields sure to be set, by stage name.

    Example:
        >>> fields_set_before(dense_graph, writes)["retrieve"]
        frozenset({'conversational_query', ..., 'query_embedding'})
    """
    sources = _sources_by_stage(graph)
    set_before = dict.fromkeys(graph.stages, REQUEST_FIELDS.union(*writes.values()))
    # Each pass keeps what is set after every source of a stage. It only ever
    # removes fields, so it stops, at what is set on every path, cycles included.
    while (narrowed := _narrowed(set_before, sources, writes)) != set_before:
        set_before = narrowed
    return set_before


# --- stages --------------------------------------------------------------------------


def _check_stage_names(graph: GraphDescription) -> None:
    for name in graph.stages:
        if name in _RESERVED_STAGE_NAMES:
            raise WorkflowGraphError(
                f"stage {name!r} is named like {_RESERVED_STAGE_NAMES[name]} "
                f"(expected a name other than {_quoted(_RESERVED_STAGE_NAMES)})"
            )


def _declared_fields(
    name: str,
    stage: StageDescription,
    components: Mapping[str, str],
    component_types: Mapping[str, type[Any]],
    stage_types: StageTypes,
) -> DeclaredFields:
    stage_type = _stage_type(name, stage, stage_types)
    settings = _stage_settings(name, stage, stage_type)
    _check_call(name, stage, stage_type, components, component_types)
    return DeclaredFields(
        stage_type.fields_read(settings), stage_type.fields_written(settings)
    )


def _stage_type(
    name: str, stage: StageDescription, stage_types: StageTypes
) -> type[BaseStage[Any, Any]]:
    try:
        return stage_types.get(stage.type_name(name))
    except ValueError as exc:  # the registry names the known stage types
        raise WorkflowGraphError(f"stage {name!r}: {exc}") from exc


def _stage_settings(
    name: str, stage: StageDescription, stage_type: type[BaseStage[Any, Any]]
) -> StageSettings:
    try:
        return stage_type.settings_type.model_validate(stage.stage_settings)
    except ValidationError as exc:
        problems = "; ".join(_problem(e["loc"], e["msg"]) for e in exc.errors())
        keys = _quoted(stage_type.settings_type.model_fields)
        expected = f"its settings among: {keys}" if keys else "no settings"
        raise WorkflowGraphError(
            f"stage {name!r} has settings its stage type "
            f"{stage.type_name(name)!r} refuses: {problems} (expected {expected})"
        ) from exc


def _problem(where: tuple[int | str, ...], what: str) -> str:
    # One problem pydantic found, on one line: where (none for the whole
    # settings, such as a user prompt the stage type cannot fill), then what.
    path = ".".join(str(part) for part in where)
    return f"{path}: {what}" if path else what


def _check_call(
    name: str,
    stage: StageDescription,
    stage_type: type[BaseStage[Any, Any]],
    components: Mapping[str, str],
    component_types: Mapping[str, type[Any]],
) -> None:
    if stage.call not in components:
        raise WorkflowGraphError(
            f"stage {name!r} calls {stage.call!r}, which is no component "
            f"(expected one of: {_quoted(components)})"
        )
    needed = stage_type.component_type
    port = component_types.get(components[stage.call])
    if port is not None and issubclass(port, needed):
        return
    fitting = [
        kind
        for kind, kind_port in component_types.items()
        if issubclass(kind_port, needed)
    ]
    raise WorkflowGraphError(
        f"stage {name!r} calls {stage.call!r}, a component of type "
        f"{components[stage.call]!r}, but its stage type {stage.type_name(name)!r} "
        f"calls one of type {_quoted(fitting) or needed.__name__} "
        f"({_components_of(fitting, components)})"
    )


def _components_of(fitting: list[str], components: Mapping[str, str]) -> str:
    # What the author may write in call instead.
    names = [name for name, kind in components.items() if kind in fitting]
    if not names:
        return "expected a component of that type, and none is declared"
    return f"expected one of: {_quoted(names)}"


# --- edges ---------------------------------------------------------------------------


def _check_edges(graph: GraphDescription) -> None:
    for source, target in graph.edge_pairs():
        edge = f"edge {source!r} -> {target!r}"
        _check_source(graph, source, edge)
        _check_target(graph, target, edge)


def _check_source(graph: GraphDescription, source: str, edge: str) -> None:
    if source != START and source not in graph.stages:
        raise WorkflowGraphError(
            f"{edge} starts at {source!r}, which is no stage "
            f"(expected START or one of the stages: {_quoted(graph.stages)})"
        )


def _check_target(graph: GraphDescription, target: str, edge: str) -> None:
    if target != END and target not in graph.stages:
        raise WorkflowGraphError(
            f"{edge} leads to {target!r}, which is no stage "
            f"(expected END or one of the stages: {_quoted(graph.stages)})"
        )


def _check_conditional_edges(
    graph: GraphDescription, edge_conditions: Registry[EdgeCondition]
) -> None:
    asked: set[tuple[str, str]] = set()
    for edge in graph.conditional_edges:
        _check_conditional_edge(graph, edge, edge_conditions)
        if (edge.source, edge.path) in asked:
            # LangGraph would refuse it too, naming the branch, not the file.
            raise WorkflowGraphError(
                f"edge condition {edge.path!r} is used twice on {edge.source!r} "
                "(expected each edge condition once per source: LangGraph names a "
                "branch after its edge condition, and one conditional edge "
                "already says where its true and its false answer lead)"
            )
        asked.add((edge.source, edge.path))


def _check_conditional_edge(
    graph: GraphDescription,
    edge: ConditionalEdge,
    edge_conditions: Registry[EdgeCondition],
) -> None:
    where = f"conditional edge from {edge.source!r} on {edge.path!r}"
    _check_source(graph, edge.source, where)
    try:
        edge_conditions.get(edge.path)
    except ValueError as exc:  # the registry names the known edge conditions
        raise WorkflowGraphError(f"{where}: {exc}") from exc
    for answer in (True, False):
        said = str(answer).lower()
        if answer not in edge.path_map:
            raise WorkflowGraphError(
                f"{where} has no {said} target (expected a path_map with a true "
                "and a false target, each a stage or END)"
            )
        _check_target(graph, edge.path_map[answer], f"the {said} answer of the {where}")


def _check_entry(graph: GraphDescription) -> None:
    if all(source != START for source, _ in _links(graph)):
        raise WorkflowGraphError(
            "the graph has no edge from START (expected a chain of plain edges "
            "that starts with START, or a conditional edge from START)"
        )


def _links(graph: GraphDescription) -> list[tuple[str, str]]:
    # Every way from one stage to the next: each plain edge, and each answer of
    # a conditional edge.
    answers = [
        (edge.source, target)
        for edge in graph.conditional_edges
        for target in edge.path_map.values()
    ]
    return [*graph.edge_pairs(), *answers]


# --- the field rule ------------------------------------------------------------------


def _check_fields_read(
    graph: GraphDescription, fields: Mapping[str, DeclaredFields]
) -> None:
    writes = {name: declared.writes for name, declared in fields.items()}
    set_before = fields_set_before(graph, writes)
    for name, declared in fields.items():
        unset = sorted(declared.reads - set_before[name])
        if unset:
            raise WorkflowGraphError(_unset_field_message(name, unset[0], writes))


def _unset_field_message(
    stage: str, field: str, writes: Mapping[str, frozenset[str]]
) -> str:
    writers = [name for name, written in writes.items() if field in written]
    written = (
        f"is written by {_quoted(writers)}, but not on every path from START "
        f"to {stage!r}"
        if writers
        else "is written by no stage of the graph"
    )
    return (
        f"stage {stage!r} reads the workflow state field {field!r}, which the "
        f"request does not provide and {written} (expected a field the request "
        f"provides, one of: {_quoted(REQUEST_FIELDS)}, or one a stage writes on "
        f"every path from START to {stage!r})"
    )


def _sources_by_stage(graph: GraphDescription) -> dict[str, set[str]]:
    sources: dict[str, set[str]] = {stage: set() for stage in graph.stages}
    for source, target in _links(graph):
        if target in sources:  # END runs nothing
            sources[target].add(source)
    return sources


def _narrowed(
    set_before: Mapping[str, frozenset[str]],
    sources: Mapping[str, set[str]],
    writes: Mapping[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    def set_after(source: str) -> frozenset[str]:
        if source == START:
            return REQUEST_FIELDS
        return set_before[source] | writes[source]

    return {
        stage: fields.intersection(*map(set_after, sources[stage]))
        for stage, fields in set_before.items()
    }


def _quoted(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in sorted(names))
