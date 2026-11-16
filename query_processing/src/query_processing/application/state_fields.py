"""The workflow state's fields and their types, built from the workflow file.

A field exists because the request or a stage writes it, and its type is the
type of what writes it (spec 3.2). Two writers that disagree on a field's type
are refused; a writer of type ``Any`` is not compared. LangGraph is handed a
``TypedDict`` of those fields, built at run time, which always holds
``timings_ms`` with its reducer. ``metadata_filter`` accumulates: a stage
that writes it adds conditions and never removes the client's.

Example:
    >>> types = field_types_of({"embed": FieldTypes(
    ...     {"standalone_query": str}, {"query_embedding": list[float]})})
    >>> StateGraph(state_type(types))
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, NamedTuple, TypedDict, cast

from ..domain.metadata_filter import MetadataCondition
from .stages.base_stage import FieldTypes, type_name
from .workflow_state import (
    METADATA_FILTER_FIELD,
    REQUEST_FIELD_TYPES,
    TIMINGS_FIELD,
    merge_timings,
)

# Who writes the request's fields, as a refusal names it.
REQUEST_WRITER = "the request"


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


def state_type(field_types: Mapping[str, Any]) -> type[Any]:
    """Return the LangGraph state of these fields, with ``timings_ms`` and its reducer.

    Args:
        field_types: Each field's type, by name, as ``field_types_of`` gives.

    Returns:
        A ``TypedDict`` in which no field is required: a field is set when its
        writer has run. ``metadata_filter`` and ``timings_ms`` are always
        there, with their reducers.

    Example:
        >>> get_type_hints(state_type({"draft": str}))["draft"]
        <class 'str'>
    """
    fields = {
        **field_types,
        METADATA_FILTER_FIELD: Annotated[list[MetadataCondition], add_conditions],
        TIMINGS_FIELD: Annotated[dict[str, float], merge_timings],
    }
    # The functional form, since the fields are known only at run time; mypy
    # types only a literal dict there.
    make_typed_dict = cast(Any, TypedDict)
    return cast(type[Any], make_typed_dict("WorkflowState", fields, total=False))


def add_conditions(
    left: Sequence[MetadataCondition], right: Sequence[MetadataCondition]
) -> list[MetadataCondition]:
    """Add a stage's conditions to the metadata filter; the field's LangGraph reducer.

    Args:
        left: The conditions so far, the client's first.
        right: The conditions a stage writes.

    Returns:
        ``left``, then each condition of ``right`` it does not hold yet.

    Example:
        >>> add_conditions([on_pay], [on_pay, not_on_leave])
        [on_pay, not_on_leave]
    """
    added = list(left)
    for condition in right:
        if condition not in added:
            added.append(condition)
    return added
