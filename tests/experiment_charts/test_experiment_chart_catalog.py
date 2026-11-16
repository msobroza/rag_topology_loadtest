"""The catalog: eleven charts, numbered in order, each with a title."""

from __future__ import annotations

from pathlib import Path

from experiment_chart_support import REGISTRY_DIR
from experiment_charts.chart_catalog import experiment_charts
from experiment_charts.write_experiment_charts import main


def test_eleven_charts_with_unique_slugs_in_order() -> None:
    slugs = [chart.slug for chart in experiment_charts()]
    assert slugs == [
        "01-where-the-time-goes",
        "02-more-users-more-waiting",
        "03-bigger-machine-more-answers",
        "04-scale-out-or-scale-up",
        "05-same-weights-same-throughput",
        "06-who-waits-how-long",
        "07-one-request-fills-the-machine",
        "08-small-batches",
        "09-overload",
        "10-topologies",
        "11-larger-model",
    ]
    assert all(chart.title for chart in experiment_charts())


def test_main_writes_eleven_tables_and_twenty_two_figures(tmp_path: Path) -> None:
    argv = ["--registry", str(REGISTRY_DIR), "--out", str(tmp_path)]
    assert main([*argv, "--image-format", "png"]) == 0
    assert len(list((tmp_path / "data").glob("*.csv"))) == 11
    assert len(list((tmp_path / "figures").glob("*.png"))) == 22
