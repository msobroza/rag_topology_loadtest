"""Chart 5: the median and the 95th-percentile wait, one row per configuration.

Example:
    >>> WhoWaitsHowLongChart(panels, users=16).table(registry).column("p95_s")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .base_experiment_chart import BaseExperimentChart
from .chart_table import ChartTable
from .chart_theme import (
    ChartTheme,
    axis_labels,
    figure_title,
    panel_title,
    themed_figure,
)
from .registry_reading import ExperimentRegistry


@dataclass(frozen=True)
class ServerRun:
    """One row of a dumbbell panel: its label, its model server, its experiment."""

    label: str
    server: str  # "FastAPI" or "OVMS"
    experiment: str


@dataclass(frozen=True)
class DumbbellPanel:
    """One panel: a machine size and its configurations, top to bottom."""

    title: str
    rows: tuple[ServerRun, ...]


@dataclass(frozen=True)
class WhoWaitsHowLongChart(BaseExperimentChart):
    """Horizontal dumbbells from the median to the 95th percentile, per panel."""

    panels: tuple[DumbbellPanel, ...]
    users: int
    slug: ClassVar[str] = "05-who-waits-how-long"
    title: ClassVar[str] = (
        "How long requests wait with 16 users at once: "
        "the median and the slowest 1 in 20"
    )
    median_label: ClassVar[str] = "half of the requests wait less than this"
    slowest_label: ClassVar[str] = "1 request in 20 waits longer than this"

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Each configuration's median and 95th-percentile wait, in seconds.

        Args:
            registry: The experiment registry.

        Returns:
            One row per dumbbell: panel, label, server, experiment, users, median_s,
            p95_s.

        Example:
            >>> chart.table(registry).column("median_s")[:2]
            (15.0, 13.0)
        """
        rows = []
        for panel in self.panels:
            for run in panel.rows:
                route = registry.route_row(run.experiment, self.users)
                rows.append(
                    (
                        panel.title,
                        run.label,
                        run.server,
                        run.experiment,
                        self.users,
                        route.p50_ms / 1000,
                        route.p95_ms / 1000,
                    )
                )
        columns = ("panel", "label", "server", "experiment", "users")
        return ChartTable((*columns, "median_s", "p95_s"), tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Two panels sharing the x axis in seconds, a legend for the two dots.

        Args:
            table: What ``table`` returned.
            theme: The light or the dark theme.

        Returns:
            The figure.

        Example:
            >>> chart.draw(chart.table(registry), dark_theme())
        """
        figure, axes = themed_figure(theme, panels=2, share_x=True)
        figure_title(figure, theme, self.title)
        for each, panel in zip(axes, table.distinct("panel"), strict=True):
            _draw_dumbbells(each, theme, table.rows_where("panel", panel))
            panel_title(each, theme, str(panel))
            axis_labels(each, theme, "seconds", "")
        axes[0].set_xlim(left=0)
        figure.legend(
            handles=self._legend_handles(theme),
            loc="outside lower center",
            ncols=2,
            frameon=False,
            labelcolor=theme.ink,
            fontsize=theme.text_size,
        )
        return figure

    def _legend_handles(self, theme: ChartTheme) -> list[Line2D]:
        shapes = [
            _dot(theme, theme.ink_secondary, hollow=False, label=self.median_label),
            _dot(theme, theme.ink_secondary, hollow=True, label=self.slowest_label),
        ]
        servers = [_dot(theme, theme.fastapi, False, "FastAPI model server")]
        return [*shapes, *servers, _dot(theme, theme.ovms, False, "OVMS")]


def _draw_dumbbells(axes: Axes, theme: ChartTheme, panel: ChartTable) -> None:
    rows = range(len(panel.rows))
    medians, slowest = panel.numbers("median_s"), panel.numbers("p95_s")
    for row, server, median, p95 in zip(
        rows, panel.column("server"), medians, slowest, strict=True
    ):
        colour = theme.server_colour(str(server))
        axes.plot([median, p95], [row, row], color=colour, linewidth=theme.line_width)
        # The median on top: it can sit right under the 95th percentile's diamond.
        axes.add_line(_dot(theme, colour, True, None, (p95, row)))
        axes.add_line(_dot(theme, colour, False, None, (median, row)))
    axes.set_yticks(rows, [str(label) for label in panel.column("label")])
    axes.set_ylim(len(panel.rows) - 0.5, -0.5)  # the first row on top
    axes.yaxis.grid(False)  # a grid line would run through each dumbbell
    axes.tick_params(axis="y", length=0)


def _dot(
    theme: ChartTheme,
    colour: str,
    hollow: bool,
    label: str | None,
    at: tuple[float, float] = (0.0, 0.0),
) -> Line2D:
    # The median is a filled circle, the 95th percentile a hollow diamond: shape,
    # not colour, tells them apart, as colour already says the server.
    return Line2D(
        [at[0]],
        [at[1]],
        linestyle="none",
        marker="D" if hollow else "o",
        markersize=theme.marker_size,
        markerfacecolor=theme.surface if hollow else colour,
        markeredgecolor=colour if hollow else theme.surface,
        markeredgewidth=theme.marker_ring,
        label=label,
    )
