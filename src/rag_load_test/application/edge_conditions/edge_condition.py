"""What an edge condition is, and ``EDGE_CONDITIONS``, the registry they register in.

An edge condition is a yes-or-no question about the workflow state, asked by a
conditional edge: a function of the state alone that returns a bool. It makes
no model call and takes no other argument. LangGraph names a branch after its
function, so an edge condition is a named function, never a lambda: Domino
tracing fails on a step with no name.

Example:
    >>> @EDGE_CONDITIONS.register("is_query_mode")
    ... def is_query_mode(state: WorkflowState) -> bool: ...
    >>> EDGE_CONDITIONS.get("is_query_mode")({"mode": "retrieve"})
    False
"""

from __future__ import annotations

from collections.abc import Callable

from ...domain.registry import Registry
from ..workflow_state import WorkflowState

EdgeCondition = Callable[[WorkflowState], bool]

EDGE_CONDITIONS: Registry[EdgeCondition] = Registry("edge condition")
