"""``ChartTable``: its columns, its filters and its CSV."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiment_charts.chart_table import ChartTable

TABLE = ChartTable(
    ("group", "server", "requests_per_s"),
    (("8 vCPUs, 32-bit", "FastAPI", 1.0044), ("8 vCPUs, 32-bit", "OVMS", 2.0)),
)


def test_the_csv_has_the_columns_then_one_line_a_row(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    TABLE.write_csv(path)
    assert path.read_text(encoding="utf-8") == (
        "group,server,requests_per_s\n"
        '"8 vCPUs, 32-bit",FastAPI,1.004\n'
        '"8 vCPUs, 32-bit",OVMS,2\n'
    )


def test_a_column_a_filter_and_the_distinct_values() -> None:
    assert TABLE.column("server") == ("FastAPI", "OVMS")
    assert TABLE.numbers("requests_per_s") == (1.0044, 2.0)
    assert TABLE.rows_where("server", "OVMS").rows == (TABLE.rows[1],)
    assert TABLE.distinct("group") == ("8 vCPUs, 32-bit",)


def test_an_unknown_column_names_the_columns() -> None:
    with pytest.raises(KeyError, match="'users'"):
        TABLE.column("users")
