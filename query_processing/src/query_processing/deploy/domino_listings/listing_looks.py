"""The look a listing takes: the list in a terminal, the table, or the details.

Example:
    >>> chosen_look(details=False, table=False, terminal=False)
    'table'
"""

from __future__ import annotations

from typing import Literal

Look = Literal["list", "table", "details"]


def chosen_look(*, details: bool, table: bool, terminal: bool) -> Look:
    """Choose the look: what was asked, else the list in a terminal, else the table.

    Args:
        details: ``--details`` was given.
        table: ``--table`` was given.
        terminal: Both the input and the output are a terminal.

    Returns:
        ``"details"`` when asked, else ``"table"`` when asked or when there is no
        terminal (a pipe, a file, a script), else ``"list"``.

    Example:
        >>> chosen_look(details=False, table=False, terminal=True)
        'list'
    """
    if details:
        return "details"
    if table or not terminal:
        return "table"
    return "list"
