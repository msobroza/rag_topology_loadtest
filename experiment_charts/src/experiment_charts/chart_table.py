"""The table behind a chart: one row per mark, the numbers as plotted.

Example:
    >>> ChartTable(("server", "requests_per_s"), (("FastAPI", 1.004),)).column("server")
    ('FastAPI',)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

Cell = str | int | float


@dataclass(frozen=True)
class ChartTable:
    """A tidy table: named columns, one row per mark of the chart."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Cell, ...], ...]

    def column(self, name: str) -> tuple[Cell, ...]:
        """Every row's value in the column ``name``.

        Args:
            name: A column name, e.g. ``"requests_per_s"``.

        Returns:
            The values, in row order.

        Raises:
            KeyError: The table has no such column.

        Example:
            >>> table.column("requests_per_s")
            (1.004, 1.808)
        """
        if name not in self.columns:
            raise KeyError(f"no column {name!r}; the columns are {self.columns}")
        index = self.columns.index(name)
        return tuple(row[index] for row in self.rows)

    def numbers(self, name: str) -> tuple[float, ...]:
        """The column ``name`` as floats, for plotting.

        Args:
            name: A column of numbers, e.g. ``"requests_per_s"``.

        Returns:
            The values, in row order.

        Example:
            >>> table.numbers("users")
            (1.0, 4.0)
        """
        return tuple(float(value) for value in self.column(name))

    def rows_where(self, name: str, value: Cell) -> ChartTable:
        """The rows whose column ``name`` holds ``value``, as a table.

        Args:
            name: A column name, e.g. ``"panel"``.
            value: The value to keep.

        Returns:
            A table with the same columns and the matching rows, in order.

        Example:
            >>> table.rows_where("server", "OVMS").column("experiment")
            ('v8-alone-ovms-fp32-cal4',)
        """
        kept = tuple(
            row
            for row, cell in zip(self.rows, self.column(name), strict=True)
            if cell == value
        )
        return ChartTable(self.columns, kept)

    def distinct(self, name: str) -> tuple[Cell, ...]:
        """The values of the column ``name``, each once, in order of appearance.

        Args:
            name: A column name, e.g. ``"panel"``.

        Returns:
            The distinct values.

        Example:
            >>> table.distinct("server")
            ('FastAPI', 'OVMS')
        """
        return tuple(dict.fromkeys(self.column(name)))

    def write_csv(self, path: Path) -> None:
        """Write the table as CSV, floats with at most 3 decimals.

        Args:
            path: The file to write; its folder must exist.

        Example:
            >>> table.write_csv(Path("experiments-explained/data/03-bigger.csv"))
        """
        with path.open("w", encoding="utf-8", newline="") as written:
            writer = csv.writer(written, lineterminator="\n")
            writer.writerow(self.columns)
            writer.writerows([_cell_text(cell) for cell in row] for row in self.rows)


def _cell_text(cell: Cell) -> str:
    if isinstance(cell, float):
        return f"{cell:.3f}".rstrip("0").rstrip(".")
    return str(cell)
