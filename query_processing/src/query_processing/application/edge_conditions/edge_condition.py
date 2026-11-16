"""What an edge condition is, and ``EDGE_CONDITIONS``, the registry they register in.

An edge condition is a question a conditional edge asks of the workflow state
fields it reads (ADR 0007): a function of their values, in the order of the
edge's ``reads``, that answers yes or no (``-> bool``) or a label (``-> str``).
A label edge's ``path_map`` has one row for each label of the field it reads.
It makes no model call. LangGraph names a branch after its function, so an
edge condition is a named function, never a lambda: Domino tracing fails on a
step with no name.

Example:
    >>> @EDGE_CONDITIONS.register("is_query_mode")
    ... def is_query_mode(mode: str) -> bool: ...
    >>> EDGE_CONDITIONS.get("is_query_mode")("retrieve")
    False
"""

from __future__ import annotations

from collections.abc import Callable
from typing import get_type_hints

from ...domain.registry import Registry

EdgeCondition = Callable[..., bool | str]

EDGE_CONDITIONS: Registry[EdgeCondition] = Registry("edge condition")


def answers_with_label(condition: EdgeCondition) -> bool:
    """Whether an edge condition answers with a label rather than yes or no.

    Args:
        condition: A registered edge condition, its return type annotated.

    Returns:
        True when it is annotated to return a ``str``, False for a ``bool``.

    Example:
        >>> answers_with_label(EDGE_CONDITIONS.get("label"))
        True
    """
    return get_type_hints(condition).get("return") is str
