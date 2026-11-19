"""What an edge condition is, and ``EDGE_CONDITIONS``, the registry they register in.

An edge condition is a question a conditional edge asks of the workflow state
fields it reads (ADR 0007): a function of their values, in the order of the
edge's ``reads``, that answers yes or no (``-> bool``) or a label (``-> str``,
a ``Literal`` of strings included). A label edge's ``path_map`` has one row
for each label of the field it reads. Start-up refuses one that states neither
answer, or whose annotations cannot be read, and an edge that reads a field of
another type than a parameter's annotation. It makes no model call. On
LangGraph, the default orchestrator, a branch is named after its function, so
an edge condition is a named function, never a lambda: Domino tracing fails on
a step with no name.

Example:
    >>> @EDGE_CONDITIONS.register("is_query_mode")
    ... def is_query_mode(mode: str) -> bool: ...
    >>> EDGE_CONDITIONS.get("is_query_mode")("retrieve")
    False
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Literal, get_args, get_origin, get_type_hints

from ...domain.errors import describe_error
from ...domain.registry import Registry
from ..stages.base_stage import type_name

EdgeCondition = Callable[..., bool | str]

EDGE_CONDITIONS: Registry[EdgeCondition] = Registry("edge condition")

# The kinds of parameter a field's value is handed to, by position.
POSITIONAL_PARAMETERS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def answers_with_label(condition: EdgeCondition) -> bool:
    """Whether an edge condition answers with a label rather than yes or no.

    Args:
        condition: A registered edge condition, its return type annotated.

    Returns:
        True when it is annotated to return a ``str`` or a ``Literal`` of
        strings, False for a ``bool``.

    Raises:
        ValueError: It states no return type, another one, or annotations
            that cannot be read; the message says what to write.

    Example:
        >>> answers_with_label(EDGE_CONDITIONS.get("label"))
        True
    """
    answer = _hints(condition).get("return")
    if answer is bool:
        return False
    if answer is str or _is_literal_of_strings(answer):
        return True
    said = (
        "states no return type"
        if answer is None
        else f"is annotated to return {type_name(answer)}"
    )
    raise ValueError(
        f"the edge condition {_name(condition)!r} {said} (expected -> bool for a "
        "yes-or-no answer, or -> str for a label, a Literal of strings included)"
    )


def taken_types(condition: EdgeCondition) -> list[Any]:
    """Return the type each positional parameter of an edge condition takes.

    Args:
        condition: A registered edge condition.

    Returns:
        One type per positional parameter, in order: its annotation, or
        ``Any`` for one that has none.

    Raises:
        ValueError: Its annotations cannot be read.

    Example:
        >>> taken_types(EDGE_CONDITIONS.get("is_query_mode"))
        [<class 'str'>]
    """
    hints = _hints(condition)
    parameters = inspect.signature(condition).parameters.values()
    positional = [p for p in parameters if p.kind in POSITIONAL_PARAMETERS]
    return [hints.get(p.name, Any) for p in positional]


def _hints(condition: EdgeCondition) -> dict[str, Any]:
    # A name imported only under TYPE_CHECKING is undefined at run time.
    try:
        return get_type_hints(condition)
    except Exception as exc:  # any error of evaluating the annotations
        raise ValueError(
            f"the annotations of the edge condition {_name(condition)!r} cannot be "
            f"read: {describe_error(exc)} (expected annotations of names the "
            "module imports at run time, not under TYPE_CHECKING)"
        ) from exc


def _is_literal_of_strings(answer: object) -> bool:
    values = get_args(answer)
    return get_origin(answer) is Literal and all(isinstance(v, str) for v in values)


def _name(condition: EdgeCondition) -> str:
    return getattr(condition, "__name__", repr(condition))
