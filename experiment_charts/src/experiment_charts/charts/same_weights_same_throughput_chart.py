"""Chart 5: the two model servers on the same weights, one reranker alone.

Example:
    >>> SameWeightsSameThroughputChart(groups, users=4).table(registry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.figure import Figure

from ..base_experiment_chart import BaseExperimentChart
from ..chart_marks import server_colours, table_bars
from ..chart_selections import SERVER_PAIR_COLUMNS, RunPair, server_pair_rows
from ..chart_table import ChartTable
from ..chart_theme import (
    ChartTheme,
    add_legend,
    axis_labels,
    figure_title,
    themed_figure,
)
from ..registry_reading import ExperimentRegistry


@dataclass(frozen=True)
class SameWeightsSameThroughputChart(BaseExperimentChart):
    """Grouped bars, FastAPI then OVMS, one group per machine and weights."""

    groups: tuple[RunPair, ...]
    users: int
    slug: ClassVar[str] = "05-same-weights-same-throughput"
    title: ClassVar[str] = (
        "The two model servers on the same weights (one reranker, 4 users at once)"
    )

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Each server's requests a second in each group.

        Args:
            registry: The experiment registry.

        Returns:
            One row per bar: group, server, experiment, users, requests_per_s.

        Example:
            >>> chart.table(registry).column("requests_per_s")[:2]
            (0.235, 0.238)
        """
        rows = server_pair_rows(registry, self.groups, self.users)
        return ChartTable(SERVER_PAIR_COLUMNS, tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Twelve bars in six groups, no value printed.

        Args:
            table: What ``table`` returned.
            theme: The light or the dark theme.

        Returns:
            The figure.

        Example:
            >>> chart.draw(chart.table(registry), light_theme())
        """
        figure, (axes,) = themed_figure(theme, panels=1)
        figure_title(figure, theme, self.title)
        columns = ("group", "server", "requests_per_s")
        table_bars(axes, theme, table, columns, server_colours(theme))
        axis_labels(axes, theme, "", "requests a second")
        add_legend(axes, theme, "upper left")
        return figure
