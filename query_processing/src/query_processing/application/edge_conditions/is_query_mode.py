"""The edge condition ``is_query_mode``: whether the request runs every stage.

A ``/retrieve`` request stops after rerank: its conditional edge sends a
request in ``query`` mode on to generate and any other to the end.

Example:
    >>> is_query_mode({"mode": "retrieve"})
    False
"""

from __future__ import annotations

from ..workflow_state import WorkflowState
from .edge_condition import EDGE_CONDITIONS


@EDGE_CONDITIONS.register("is_query_mode")
def is_query_mode(state: WorkflowState) -> bool:
    """Whether the request is in ``query`` mode, the one that generates an answer.

    Args:
        state: The workflow state; only its mode is read.

    Returns:
        True in ``query`` mode, False otherwise.

    Example:
        >>> is_query_mode({"mode": "query"})
        True
    """
    return state.get("mode") == "query"
