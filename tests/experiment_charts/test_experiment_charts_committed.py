"""The committed tables of ``experiments-explained/`` are what the registry gives now.

A registry change without ``make experiment-charts`` fails here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiment_chart_support import EXPLAINED_DIR, real_registry
from experiment_charts.base_experiment_chart import BaseExperimentChart
from experiment_charts.chart_catalog import experiment_charts


@pytest.mark.parametrize("chart", experiment_charts(), ids=lambda chart: chart.slug)
def test_the_committed_table_is_current(
    chart: BaseExperimentChart, tmp_path: Path
) -> None:
    written = tmp_path / f"{chart.slug}.csv"
    chart.table(real_registry()).write_csv(written)
    committed = EXPLAINED_DIR / "data" / f"{chart.slug}.csv"
    assert committed.read_text(encoding="utf-8") == written.read_text(
        encoding="utf-8"
    ), "run make experiment-charts"


def test_both_figures_of_each_chart_are_committed() -> None:
    figures = EXPLAINED_DIR / "figures"
    expected = {
        f"{chart.slug}{suffix}.svg"
        for chart in experiment_charts()
        for suffix in ("", "-dark")
    }
    assert {path.name for path in figures.glob("*.svg")} == expected
