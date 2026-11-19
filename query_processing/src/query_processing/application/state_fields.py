"""The workflow state's fields and their types, built from the workflow file.

A field exists because the request or a stage writes it, and its type is the
type of what writes it (spec 3.2). Two writers that disagree on a field's type
are refused; a writer of type ``Any`` is not compared. The orchestrator is
handed those fields, which always hold ``timings_ms`` and ``tokens``, and the
reducer of each field that adds up. ``metadata_filter`` accumulates: a stage
that writes it adds conditions and never removes the client's.

Example:
    >>> types = field_types_of({"embed": FieldTypes(
    ...     {"standalone_query": str}, {"query_embedding": list[float]})})
    >>> state_field_types(types)["timings_ms"]
    dict[str, float]
"""

from __future__ import annotations

from collections.abc import Mapping
from types import UnionType
from typing import Any, Literal, NamedTuple, Union, get_args, get_origin

from ..domain.opened_graph import Reducer
from ..domain.retrieval.metadata_filter import MetadataCondition
from .stages.base_stage import FieldTypes, type_name
from .workflow_state import (
    METADATA_FILTER_FIELD,
    REQUEST_FIELD_TYPES,
    TIMINGS_FIELD,
    TOKENS_FIELD,
    StageTokens,
    add_conditions,
    sum_timings_by_stage,
    sum_tokens_by_stage,
)

# Who writes the request's fields, as a refusal names it.
REQUEST_WRITER = "the request"
# The origins of a union: ``X | None`` and ``Optional[X]`` are told apart.
_UNIONS = (Union, UnionType)


class WorkflowGraphError(ValueError):
    """A workflow graph that cannot run: the message says what and what was expected."""


class WrittenField(NamedTuple):
    """A field's type, and the writer that gave it, as a refusal names it.

    Example:
        >>> WrittenField(str, "the request").writer
        'the request'
    """

    type: Any
    writer: str


def field_types_of(stages: Mapping[str, FieldTypes]) -> dict[str, Any]:
    """Return the type of each field the request and the stages write.

    Args:
        stages: What each stage reads and writes, with types, by stage name.

    Returns:
        Each field written, by the request or a stage, with its type; ``Any``
        only when every writer writes it as ``Any``.

    Raises:
        WorkflowGraphError: Two writers write one field as two types; the
            message names both.

    Example:
        >>> field_types_of({"draft": FieldTypes({}, {"draft": str})})["draft"]
        <class 'str'>
    """
    return {field: entry.type for field, entry in written_fields(stages).items()}


def written_fields(stages: Mapping[str, FieldTypes]) -> dict[str, WrittenField]:
    """Return each field the request and the stages write, its type and its writer.

    The writer is the request, else the first stage that writes the field as a
    type other than ``Any``.

    Args:
        stages: What each stage reads and writes, with types, by stage name.

    Returns:
        Each field written, with its type and the writer that gave it.

    Raises:
        WorkflowGraphError: Two writers write one field as two types.

    Example:
        >>> written_fields({"draft": FieldTypes({}, {"draft": str})})["draft"]
        WrittenField(type=<class 'str'>, writer="'draft'")
    """
    written = {
        field: WrittenField(kind, REQUEST_WRITER)
        for field, kind in REQUEST_FIELD_TYPES.items()
    }
    for stage, types in stages.items():
        for field, kind in types.writes.items():
            then = WrittenField(kind, repr(stage))
            written[field] = _agreed(field, written.get(field), then)
    return written


def _agreed(field: str, first: WrittenField | None, then: WrittenField) -> WrittenField:
    if first is None or first.type is Any:
        return then
    if then.type is Any or then.type == first.type:
        return first
    raise WorkflowGraphError(
        f"{field!r} is written as {type_name(first.type)} by {first.writer} and "
        f"as {type_name(then.type)} by {then.writer} (expected every writer of a "
        "field to write one type)"
    )


def type_fits(given: Any, takes: Any) -> bool:
    """Whether a stage or the response can read a field written as ``given``.

    A request or a result attribute takes a field of its own type, or of one
    member of its union: an attribute that accepts ``None`` reads a field of
    its other type. ``Any`` on either side is not compared. A written union
    fits only an attribute of that very union, never a wider one.

    Args:
        given: The type the field's writer writes.
        takes: The type of the attribute the field fills.

    Returns:
        True when ``given`` is ``takes`` or a member of it.

    Example:
        >>> type_fits(list[float], list[float] | None)
        True
    """
    if Any in (given, takes) or given == takes:
        return True
    return get_origin(takes) in _UNIONS and given in get_args(takes)


def annotation_accepts(given: Any, annotation: Any) -> bool:
    """Whether a function's parameter, as annotated, takes a field written as ``given``.

    An edge condition is a plain function, so its annotation may be wider
    than the field: ``object``, ``Sequence[object]`` for a list, ``str`` for
    a ``Literal`` of strings, or a union of which one member accepts it. A
    field that may hold ``None`` fits only an annotation that takes ``None``.

    Args:
        given: The type the field's writer writes.
        annotation: The parameter's annotation.

    Returns:
        True when every value of ``given`` is a value of ``annotation``.

    Example:
        >>> annotation_accepts(list[HistoryMessage], Sequence[object])
        True
    """
    if type_fits(given, annotation) or annotation is object:
        return True
    if get_origin(given) in _UNIONS:
        return all(annotation_accepts(m, annotation) for m in get_args(given))
    if get_origin(annotation) in _UNIONS:
        return any(annotation_accepts(given, m) for m in get_args(annotation))
    if get_origin(given) is Literal:
        return _literal_accepted(get_args(given), annotation)
    return _class_accepts(given, annotation)


def _literal_accepted(values: tuple[Any, ...], annotation: Any) -> bool:
    # Literal["query", "retrieve"] is accepted by str: each value is one.
    if not isinstance(annotation, type):
        return False
    return all(isinstance(value, annotation) for value in values)


def _class_accepts(given: Any, annotation: Any) -> bool:
    # A class, or a generic such as list[float], and what it holds.
    given_class = get_origin(given) or given
    taken_class = get_origin(annotation) or annotation
    if not (isinstance(given_class, type) and isinstance(taken_class, type)):
        return False
    if not issubclass(given_class, taken_class):
        return False
    given_args, taken_args = get_args(given), get_args(annotation)
    if not taken_args:
        return True
    if len(given_args) != len(taken_args):
        return False
    return all(map(annotation_accepts, given_args, taken_args))


def state_field_types(field_types: Mapping[str, Any]) -> dict[str, Any]:
    """Return every field of the workflow state with its type, the measured ones too.

    Args:
        field_types: Each field the request and the stages write, with its
            type, as ``field_types_of`` gives.

    Returns:
        Those fields, then ``metadata_filter``, ``timings_ms`` and ``tokens``
        with the types their reducers take: what the orchestrator builds the
        state from, with ``STATE_REDUCERS``.

    Example:
        >>> list(state_field_types({"draft": str}))[-2:]
        ['timings_ms', 'tokens']
    """
    return {
        **field_types,
        METADATA_FILTER_FIELD: list[MetadataCondition],
        TIMINGS_FIELD: dict[str, float],
        TOKENS_FIELD: dict[str, StageTokens],
    }


# The fields that add up, each with its reducer: a stage's update is added to
# the value so far, never put in its place.
STATE_REDUCERS: dict[str, Reducer] = {
    METADATA_FILTER_FIELD: add_conditions,
    TIMINGS_FIELD: sum_timings_by_stage,
    TOKENS_FIELD: sum_tokens_by_stage,
}
