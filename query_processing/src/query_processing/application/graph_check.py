"""``check_graph``: a workflow graph that cannot run, refused before it is built.

The checks know no stage type, no component and no edge condition. They read
the ``reads`` and ``writes`` every stage states, checked by its stage type's
``check_fields`` or, for a plain call, bound to its component's request and
answer by ``bind_call``; what each stage type declares (the port of the
component it calls and its settings); the declared components with their
types; and the registered edge conditions. All of it is
in the workflow file and the settings, so ``qp-serve`` runs them before any
component opens, and a refusal names what is wrong and what was expected. What
only an opened component shows, a store that ranks no text under a lexical
retrieval, is refused when the graph is built, as a ``WorkflowGraphError`` too.

The field rule: a stage may read a workflow state field only if the request
provides it or a stage writes it on every path from ``START`` to the stage. A
dense retrieval with no embed stage before it is refused by that rule, not by a
rule of its own (ADR 0004). A conditional edge states what it reads as a stage
does (ADR 0007), and is held to the same rule at the end of its source; the
checks of the edges themselves are in ``edge_check``.

The type rules (spec 3.4): a field's type is that of what writes it, the
request or a stage, and two writers that disagree are refused; a stage and the
response read a field only as the type it is written (an attribute that
accepts ``None`` reads its other type), and an edge as its edge condition's
parameter is annotated; and a field a stage writes is read by a stage, an edge
or the response, else it is likely a typo. A type of ``Any`` is not compared.

Example:
    >>> check_graph(settings.graph, settings.component_types(),
    ...             StageContext(settings.prompts))  # today's passes
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

from pydantic import BaseModel, ValidationError

from ..domain.base_component import COMPONENT_TYPES, BaseComponent
from ..domain.graph_description import (
    END,
    START,
    GraphDescription,
    StageDescription,
)
from ..domain.passages import TOTAL_TIMING_KEY
from ..domain.registry import Registry
from .edge_check import (
    check_edges,
    edge_name,
    edge_read_types,
    field_labels,
    links,
    quoted,
)
from .edge_conditions.edge_condition import EDGE_CONDITIONS, EdgeCondition
from .stage_binding import bind_call, stage_fields, stage_type_of
from .stages.base_stage import (
    STAGE_TYPES,
    BaseStage,
    FieldTypes,
    StageContext,
    StageSettings,
    StageTypes,
    type_name,
)
from .state_fields import WorkflowGraphError as WorkflowGraphError
from .state_fields import (
    WrittenField,
    annotation_accepts,
    type_fits,
    written_fields,
)
from .workflow_state import REQUEST_FIELDS, RESPONSE_FIELD_TYPES, RESPONSE_FIELDS

# A stage named like one of these could not be told from it.
_RESERVED_STAGE_NAMES = {
    START: "the start of the graph in its edges",
    END: "the end of the graph in its edges",
    TOTAL_TIMING_KEY: (
        "the key of the total time in timings_ms, where its timing would collide "
        "with the total"
    ),
}


# Every component type's port, by the type's name.
ComponentTypes = Registry[type[BaseComponent[Any, Any]]]


class _Declared(NamedTuple):
    # The components of the settings: each one's type and its type's keys, by
    # name, and the port of each component type.
    types: Mapping[str, str]
    type_settings: Mapping[str, BaseModel | None]
    ports: ComponentTypes


class _Read(NamedTuple):
    # One field a stage, an edge or the response reads, with the type it takes;
    # `reader` names it as a refusal does, and `fits` says whether a type
    # written fits that taken.
    reader: str
    field: str
    takes: Any
    fits: Callable[[Any, Any], bool] = type_fits


def check_graph(  # noqa: PLR0913 - three of them are the tests' seams
    graph: GraphDescription,
    components: Mapping[str, str],
    context: StageContext,
    component_settings: Mapping[str, BaseModel | None] | None = None,
    *,
    component_types: ComponentTypes = COMPONENT_TYPES,
    stage_types: StageTypes = STAGE_TYPES,
    edge_conditions: Registry[EdgeCondition] = EDGE_CONDITIONS,
) -> None:
    """Refuse a graph that cannot run, naming the first mistake found.

    Args:
        graph: The stages, the chains of plain edges and the conditional edges.
        components: Each declared component's type, by the component's name.
        context: What every stage is given, the prompts its settings name.
        component_settings: The keys of each component's type, by the
            component's name, from which a plain call gives the fields it
            writes their labels, such as a router's routes; None for none.
        component_types: The port of each component type, as the ports register
            them by the type's name.
        stage_types: Where a stage's type is found.
        edge_conditions: Where a conditional edge's ``path`` is found.

    Raises:
        WorkflowGraphError: A stage has a reserved name, an unknown stage type,
            settings or lists its stage type refuses, lists that do not bind to
            the component a plain call calls, or calls a missing component or
            one of the wrong type; an edge names no stage; a conditional edge names
            an unknown edge condition, reads another number of fields than it
            takes, has other rows than its answers (true and false, or the
            labels of the field it reads), or repeats an edge condition on one
            source; nothing leaves START; a stage or a conditional edge reads a
            field that is not sure to be set when it runs, or a label edge has
            other rows than the labels of its field; a stage reads a
            field as another type than its writer writes; two writers write a
            field as two types; or a stage writes a field nothing reads.

    Example:
        >>> check_graph(settings.graph, settings.component_types(),
        ...             StageContext(settings.prompts))
    """
    _check_stage_names(graph)
    declared = _Declared(components, component_settings or {}, component_types)
    fields = {
        name: _declared_fields(name, stage, declared, stage_types, context)
        for name, stage in graph.stages.items()
    }
    check_edges(graph, edge_conditions, field_labels(fields))
    _check_fields_read(graph, fields)
    _check_read_types(graph, fields, edge_conditions)
    _check_fields_unread(graph, fields)


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
                f"(expected a name other than {quoted(_RESERVED_STAGE_NAMES)})"
            )


def _declared_fields(
    name: str,
    stage: StageDescription,
    declared: _Declared,
    stage_types: StageTypes,
    context: StageContext,
) -> FieldTypes:
    stage_type = _stage_type(name, stage, stage_types)
    if stage_type is None:
        return _plain_call_fields(name, stage, declared)
    settings = _stage_settings(name, stage, stage_type)
    _check_call(name, stage, stage_type, declared.types, declared.ports)
    try:
        return stage_type.check_fields(settings, stage_fields(stage), context)
    except ValueError as exc:  # it shows the lists its stage type expects
        raise WorkflowGraphError(f"stage {name!r}: {exc}") from exc


def _stage_type(
    name: str, stage: StageDescription, stage_types: StageTypes
) -> type[BaseStage[Any, Any]] | None:
    try:
        return stage_type_of(name, stage, stage_types)
    except ValueError as exc:  # it names the known stage types
        raise WorkflowGraphError(f"stage {name!r}: {exc}") from exc


def _plain_call_fields(
    name: str, stage: StageDescription, declared: _Declared
) -> FieldTypes:
    # A plain call declares the fields it states, with the types of the request
    # and answer attributes they bind to, and the labels its component type
    # declares for an answer attribute, such as a router's routes.
    _check_declared(name, stage, declared.types)
    try:
        component_type = declared.ports.get(declared.types[stage.call])
        labels = component_type.answer_labels(declared.type_settings.get(stage.call))
        binding = bind_call(
            component_type, stage.stage_settings, stage_fields(stage), labels
        )
    except (ValueError, TypeError) as exc:
        raise WorkflowGraphError(f"stage {name!r}: {exc}") from exc
    return binding.field_types


def _stage_settings(
    name: str, stage: StageDescription, stage_type: type[BaseStage[Any, Any]]
) -> StageSettings:
    try:
        return stage_type.settings_type.model_validate(stage.stage_settings)
    except ValidationError as exc:
        problems = "; ".join(_problem(e["loc"], e["msg"]) for e in exc.errors())
        keys = quoted(stage_type.settings_type.model_fields)
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
    component_types: ComponentTypes,
) -> None:
    _check_declared(name, stage, components)
    needed = stage_type.component_type
    fitting = [
        kind
        for kind in component_types.names()
        if issubclass(component_types.get(kind), needed)
    ]
    if components[stage.call] in fitting:
        return
    raise WorkflowGraphError(
        f"stage {name!r} calls {stage.call!r}, a component of type "
        f"{components[stage.call]!r}, but its stage type {stage.type_name(name)!r} "
        f"calls one of type {quoted(fitting) or needed.__name__} "
        f"({_components_of(fitting, components)})"
    )


def _check_declared(
    name: str, stage: StageDescription, components: Mapping[str, str]
) -> None:
    if stage.call not in components:
        raise WorkflowGraphError(
            f"stage {name!r} calls {stage.call!r}, which is no component "
            f"(expected one of: {quoted(components)})"
        )


def _components_of(fitting: list[str], components: Mapping[str, str]) -> str:
    # What the author may write in call instead.
    names = [name for name, kind in components.items() if kind in fitting]
    if not names:
        return "expected a component of that type, and none is declared"
    return f"expected one of: {quoted(names)}"


# --- the field rule ------------------------------------------------------------------


def _check_fields_read(
    graph: GraphDescription, fields: Mapping[str, FieldTypes]
) -> None:
    writes = {name: frozenset(declared.writes) for name, declared in fields.items()}
    set_before = fields_set_before(graph, writes)
    for reader in _readers(graph, fields, set_before, writes):
        unset = sorted(set(reader.reads) - reader.set_by_then)
        if unset:
            raise WorkflowGraphError(_unset_field_message(reader, unset[0], writes))


class _Reader(NamedTuple):
    # A stage or a conditional edge, what it reads, and what is sure to be set
    # when it reads: the paths from START end at it, named in `upto`.
    name: str
    upto: str
    reads: Iterable[str]
    set_by_then: frozenset[str]


def _readers(
    graph: GraphDescription,
    fields: Mapping[str, FieldTypes],
    set_before: Mapping[str, frozenset[str]],
    writes: Mapping[str, frozenset[str]],
) -> list[_Reader]:
    # A conditional edge is asked once its source has run.
    stages = [
        _Reader(f"stage {name!r}", repr(name), declared.reads, set_before[name])
        for name, declared in fields.items()
    ]
    edges = [
        (
            _Reader(edge_name(edge), "START", edge.reads, REQUEST_FIELDS)
            if edge.source == START
            else _Reader(
                edge_name(edge),
                f"the end of {edge.source!r}",
                edge.reads,
                set_before[edge.source] | writes[edge.source],
            )
        )
        for edge in graph.conditional_edges
    ]
    return [*stages, *edges]


def _unset_field_message(
    reader: _Reader, field: str, writes: Mapping[str, frozenset[str]]
) -> str:
    writers = [name for name, written in writes.items() if field in written]
    written = (
        f"is written by {quoted(writers)}, but not on every path from START "
        f"to {reader.upto}"
        if writers
        else "is written by no stage of the graph"
    )
    return (
        f"{reader.name} reads the workflow state field {field!r}, which the "
        f"request does not provide and {written} (expected a field the request "
        f"provides, one of: {quoted(REQUEST_FIELDS)}, or one a stage writes on "
        f"every path from START to {reader.upto})"
    )


def _check_read_types(
    graph: GraphDescription,
    fields: Mapping[str, FieldTypes],
    edge_conditions: Registry[EdgeCondition],
) -> None:
    # A field's type is its writers' (written_fields refuses two that
    # disagree); a stage or the response takes it as the attribute it fills
    # does, an edge as its edge condition's parameter is annotated.
    written = written_fields(fields)
    for read in _reads(graph, fields, edge_conditions):
        given = written.get(read.field, WrittenField(Any, ""))
        if read.fits(given.type, read.takes):
            continue
        raise WorkflowGraphError(
            f"{read.reader} reads {read.field!r} as {type_name(read.takes)}, "
            f"but {given.writer} writes it as {type_name(given.type)} "
            f"(expected a field written as {type_name(read.takes)})"
        )


def _reads(
    graph: GraphDescription,
    fields: Mapping[str, FieldTypes],
    edge_conditions: Registry[EdgeCondition],
) -> list[_Read]:
    stages = [
        _Read(f"stage {stage!r}", field, takes)
        for stage, declared in fields.items()
        for field, takes in declared.reads.items()
    ]
    edges = [
        _Read(edge_name(edge), field, takes, annotation_accepts)
        for edge in graph.conditional_edges
        for field, takes in edge_read_types(edge, edge_conditions).items()
    ]
    response = [
        _Read("the response", field, takes)
        for field, takes in RESPONSE_FIELD_TYPES.items()
    ]
    return [*stages, *edges, *response]


def _check_fields_unread(
    graph: GraphDescription, fields: Mapping[str, FieldTypes]
) -> None:
    # A field no one reads is most likely a misspelt one that someone does.
    read = RESPONSE_FIELDS.union(
        *(declared.reads for declared in fields.values()),
        *(edge.reads for edge in graph.conditional_edges),
    )
    for name, declared in fields.items():
        unread = [field for field in declared.writes if field not in read]
        if unread:
            raise WorkflowGraphError(
                f"stage {name!r} writes {unread[0]!r}, but no stage, edge or "
                "response reads it, a likely typo (expected a field a stage "
                f"reads, or one the response reads: {quoted(RESPONSE_FIELDS)})"
            )


def _sources_by_stage(graph: GraphDescription) -> dict[str, set[str]]:
    sources: dict[str, set[str]] = {stage: set() for stage in graph.stages}
    for source, target in links(graph):
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
