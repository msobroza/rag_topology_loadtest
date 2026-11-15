"""The catalog: ten charts, numbered in order, each with a title."""

from __future__ import annotations

from pathlib import Path

from experiment_chart_support import REGISTRY_DIR
from experiment_charts.chart_catalog import experiment_charts
from experiment_charts.write_experiment_charts import main


def test_ten_charts_with_unique_slugs_in_order() -> None:
    slugs = [chart.slug for chart in experiment_charts()]
    assert len(slugs) == 10
    assert len(set(slugs)) == 10
    assert [slug[:2] for slug in slugs] == [f"{n:02d}" for n in range(1, 11)]
    assert all(chart.title for chart in experiment_charts())


def test_main_writes_ten_tables_and_twenty_figures(tmp_path: Path) -> None:
    argv = ["--registry", str(REGISTRY_DIR), "--out", str(tmp_path)]
    assert main([*argv, "--image-format", "png"]) == 0
    assert len(list((tmp_path / "data").glob("*.csv"))) == 10
    assert len(list((tmp_path / "figures").glob("*.png"))) == 20
