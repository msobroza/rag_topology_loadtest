"""Every chart draws in both themes, and its SVG is reproducible."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiment_chart_support import REGISTRY_DIR, real_registry
from experiment_charts.base_experiment_chart import BaseExperimentChart
from experiment_charts.chart_catalog import experiment_charts
from experiment_charts.chart_theme import ChartTheme, dark_theme, light_theme
from experiment_charts.write_experiment_charts import ChartOutput, save_figure

THEMES = [light_theme(), dark_theme()]


@pytest.mark.parametrize("theme", THEMES, ids=["light", "dark"])
@pytest.mark.parametrize("chart", experiment_charts(), ids=lambda chart: chart.slug)
def test_each_chart_draws_an_svg_without_a_date(
    chart: BaseExperimentChart, theme: ChartTheme, tmp_path: Path
) -> None:
    options = ChartOutput(REGISTRY_DIR, tmp_path, "svg")
    path = tmp_path / f"{chart.slug}.svg"
    save_figure(chart, chart.table(real_registry()), theme, path, options)
    svg = path.read_text(encoding="utf-8")
    assert svg.startswith("<?xml")
    assert "<path" in svg
    assert "<dc:date>" not in svg


def test_the_same_chart_writes_the_same_bytes_twice(tmp_path: Path) -> None:
    chart = experiment_charts()[1]
    options = ChartOutput(REGISTRY_DIR, tmp_path, "svg")
    table = chart.table(real_registry())
    first, second = tmp_path / "first.svg", tmp_path / "second.svg"
    save_figure(chart, table, light_theme(), first, options)
    save_figure(chart, table, light_theme(), second, options)
    assert first.read_bytes() == second.read_bytes()
