"""Plain text in aligned columns, as the listing commands print their tables.

Example:
    >>> print(aligned_table(["NAME", "CORES"], [["cpu-16", "16"]], right={1}), end="")
    NAME    CORES
    cpu-16     16
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

_BETWEEN = "  "  # two spaces between two columns


def aligned_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    right: Collection[int] = (),
) -> str:
    """Lay out the headers then the rows, each column padded to its widest value.

    No colour and no cursor movement, so the table reads the same in a file.

    Args:
        headers: The columns' names, the first line.
        rows: One sequence of cells a line, as many cells as headers.
        right: The indexes of the columns aligned right (numbers); the others
            are aligned left.

    Returns:
        One line a row after the headers, each ending in a newline and none
        ending in a space.

    Raises:
        ValueError: If a row has not as many cells as there are headers.

    Example:
        >>> aligned_table(["A", "B"], [["x", "1"]], right={1}).splitlines()
        ['A  B', 'x  1']
    """
    lines = [list(headers), *(list(row) for row in rows)]
    for line in lines:
        if len(line) != len(headers):
            raise ValueError(
                f"a row of the table has {len(line)} cells ({line!r}); "
                f"{len(headers)} were expected, one for each of {list(headers)!r}"
            )
    widths = [
        max(len(line[column]) for line in lines) for column in range(len(headers))
    ]
    return "".join(_laid_out(line, widths, right) + "\n" for line in lines)


def _laid_out(cells: list[str], widths: list[int], right: Collection[int]) -> str:
    padded = (
        cell.rjust(width) if column in right else cell.ljust(width)
        for column, (cell, width) in enumerate(zip(cells, widths, strict=True))
    )
    return _BETWEEN.join(padded).rstrip()
