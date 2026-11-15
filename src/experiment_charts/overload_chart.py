"""Chart 8: the share of requests that gave up, through the whole workflow.

A request gives up when the emulated Domino's proxy cuts it at 30 seconds; Locust
counts it as a failure.

Example:
    >>> OverloadChart(panels, users=(4, 16, 32)).table(registry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.figure import Figure

from .base_experiment_chart import BaseExperimentChart
from .chart_marks import label_bars, server_colours, table_bars
from .chart_selections import SERVERS, RunPair
from .chart_table import ChartTable
from .chart_theme import (
    ChartTheme,
    add_legend,
    axis_labels,
    figure_title,
    panel_title,
    themed_figure,
)
from .registry_reading import ExperimentRegistry


@dataclass(frozen=True)
class OverloadChart(BaseExperimentChart):
    """Grouped bars of the failed share by users at once, one panel per machine."""

    panels: tuple[RunPair, ...]  # a panel title, FastAPI's and OVMS's experiments
    users: tuple[int, ...]
    slug: ClassVar[str] = "08-overload"
    title: ClassVar[str] = (
        "Share of requests that gave up after 30 seconds, through the whole workflow"
    )

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Each run's requests, failures and failed percentage.

        Args:
            registry: The experiment registry.

        Returns:
            One row per bar: panel, users, server, experiment, requests, failures,
            failed_percent.

        Example:
            >>> chart.table(registry).column("failures")[:3]
            (0, 0, 0)
        """
        rows = []
        for panel in self.panels:
            runs = [
                (users, server, experiment)
                for users in self.users
                for server, experiment in zip(
                    SERVERS, (panel.first, panel.second), strict=True
                )
            ]
            for users, server, experiment in runs:
                route = registry.route_row(experiment, users)
                failed = 100 * route.failures / route.requests
                head = (panel.label, users, server, experiment)
                rows.append((*head, route.requests, route.failures, failed))
        columns = ("panel", "users", "server", "experiment", "requests", "failures")
        return ChartTable((*columns, "failed_percent"), tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Two panels from 0 to 100 %, each bar's share printed, zero as "0".

        Args:
            table: What ``table`` returned.
            theme: The light or the dark theme.

        Returns:
            The figure.

        Example:
            >>> chart.draw(chart.table(registry), dark_theme())
        """
        figure, axes = themed_figure(theme, panels=2, share_y=True)
        figure_title(figure, theme, self.title)
        columns = ("users", "server", "failed_percent")
        for each, panel in zip(axes, table.distinct("panel"), strict=True):
            runs = table.rows_where("panel", panel)
            for bars in table_bars(each, theme, runs, columns, server_colours(theme)):
                # A zero bar is invisible: its printed "0" says it was measured.
                texts = [_percent_text(bar.get_height()) for bar in bars]
                label_bars(each, theme, bars, texts)
            panel_title(each, theme, str(panel))
            axis_labels(each, theme, "users at once", "")
        axis_labels(axes[0], theme, "users at once", "% of requests that gave up")
        axes[0].set_ylim(0, 100)
        add_legend(axes[0], theme, "upper left")
        return figure


def _percent_text(percent: float) -> str:
    return f"{percent:.1f}%" if percent else "0"
