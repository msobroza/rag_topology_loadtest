"""The edge condition ``is_query_mode``: whether the request runs every stage.

A ``/retrieve`` request stops after rerank: its conditional edge, which reads
``mode``, sends a request in ``query`` mode on to generate and any other to
the end.

Example:
    >>> is_query_mode("retrieve")
    False
"""

from __future__ import annotations

from .edge_condition import EDGE_CONDITIONS


@EDGE_CONDITIONS.register("is_query_mode")
def is_query_mode(mode: str) -> bool:
    """Whether the request is in ``query`` mode, the one that generates an answer.

    Args:
        mode: The value of the field its edge reads, the request's mode.

    Returns:
        True in ``query`` mode, False otherwise.

    Example:
        >>> is_query_mode("query")
        True
    """
    return mode == "query"
