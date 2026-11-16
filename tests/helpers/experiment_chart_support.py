"""What the tests of ``experiment_charts`` share: the real registry and a chart by slug."""

from __future__ import annotations

from functools import cache

from experiment_charts.base_experiment_chart import BaseExperimentChart
from experiment_charts.chart_catalog import experiment_charts
from experiment_charts.registry_reading import (
    ExperimentRegistry,
    read_experiment_registry,
)
from repo_paths import REPO_ROOT

REGISTRY_DIR = REPO_ROOT / "experiments"
EXPLAINED_DIR = REPO_ROOT / "experiments-explained"


@cache
def real_registry() -> ExperimentRegistry:
    """The repository's registry, read once for every test."""
    return read_experiment_registry(REGISTRY_DIR)


def chart_named(slug: str) -> BaseExperimentChart:
    """The catalog's chart whose slug is ``slug``."""
    (chart,) = [chart for chart in experiment_charts() if chart.slug == slug]
    return chart
