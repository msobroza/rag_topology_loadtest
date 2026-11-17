"""The edge condition ``has_history``: whether a request carries a history.

A first message has no conversational history to lean on, so its conditional
edge, which reads the history, can send it past the router and the rewrite.

Example:
    >>> has_history([])
    False
"""

from __future__ import annotations

from collections.abc import Sequence

from .edge_condition import EDGE_CONDITIONS


@EDGE_CONDITIONS.register("has_history")
def has_history(conversational_history: Sequence[object]) -> bool:
    """Whether the conversational history holds at least one message.

    Args:
        conversational_history: The value of the field its edge reads.

    Returns:
        True when the history is not empty.

    Example:
        >>> has_history([message])  # a HistoryMessage the client sent
        True
    """
    return len(conversational_history) > 0
