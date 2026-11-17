"""The edge condition ``label``: it answers with the field it reads.

A label edge reads one field, such as the route a router writes, and its
``path_map`` has one row for each label that field's writer declares; start-up
refuses one that lists other labels (ADR 0007).

Example:
    >>> label("leans_on_history")
    'leans_on_history'
"""

from __future__ import annotations

from .edge_condition import EDGE_CONDITIONS


@EDGE_CONDITIONS.register("label")
def label(value: str) -> str:
    """Answer with the value of the field the edge reads, as it is.

    Args:
        value: The value of the one field its edge reads.

    Returns:
        ``value``, the label whose row of ``path_map`` the request takes.

    Example:
        >>> label("self_contained")
        'self_contained'
    """
    return value
