"""Chart 9: requests a second through the whole workflow, by topology.

Each bar names the reranker capacity its topology had: on four small machines the
split topologies keep fewer vCPUs for the reranker than the monolith does. The
table adds requests a second per reranker vCPU.

Example:
    >>> TopologiesChart(panels).table(registry).column("requests_per_s")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_experiment_chart import BaseExperimentChart
from .chart_marks import label_bars, rate_text, wrapped
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
class TopologyBar:
    """One topology's experiment and its reranker capacity, in words and vCPUs."""

    topology: str
    experiment: str
    reranker_capacity: str  # e.g. "4 rerankers, 2 vCPUs each"
    reranker_vcpus_total: int


@dataclass(frozen=True)
class TopologyPanel:
    """One sitting's machines: a title, the user count, its topologies in order."""

    title: str
    users: int
    bars: tuple[TopologyBar, ...]


@dataclass(frozen=True)
class TopologiesChart(BaseExperimentChart):
    """Three bars per panel, monolith then split-reranker then split-all."""

    panels: tuple[TopologyPanel, ...]
    slug: ClassVar[str] = "09-topologies"
    title: ClassVar[str] = "Requests a second through the whole workflow, by topology"

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Each topology's requests a second, and per reranker vCPU.

        Args:
            registry: The experiment registry.

        Returns:
            One row per bar: panel, topology, experiment, users, reranker_capacity,
            reranker_vcpus_total, requests_per_s, requests_per_s_per_reranker_vcpu.

        Example:
            >>> chart.table(registry).column("requests_per_s")[:3]
            (0.652, 0.676, 0.696)
        """
        rows = []
        for panel in self.panels:
            for bar in panel.bars:
                rate = registry.route_row(bar.experiment, panel.users).requests_per_s
                head = (panel.title, bar.topology, bar.experiment, panel.users)
                capacity = (bar.reranker_capacity, bar.reranker_vcpus_total)
                rows.append((*head, *capacity, rate, rate / bar.reranker_vcpus_total))
        columns = ("panel", "topology", "experiment", "users", "reranker_capacity")
        derived = ("reranker_vcpus_total", "requests_per_s")
        per_vcpu = "requests_per_s_per_reranker_vcpu"
        return ChartTable((*columns, *derived, per_vcpu), tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Two panels, each with its own y axis, every value printed.

        Args:
            table: What ``table`` returned.
            theme: The light or the dark theme.

        Returns:
            The figure.

        Example:
            >>> chart.draw(chart.table(registry), light_theme())
        """
        figure, axes = themed_figure(theme, panels=2)
        figure_title(figure, theme, self.title)
        for each, panel in zip(axes, table.distinct("panel"), strict=True):
            _draw_panel(each, theme, table.rows_where("panel", panel))
            panel_title(each, theme, str(panel))
            axis_labels(each, theme, "", "requests a second")
        return figure


def _draw_panel(axes: Axes, theme: ChartTheme, panel: ChartTable) -> None:
    positions = range(len(panel.rows))
    rates = panel.numbers("requests_per_s")
    bars = axes.bar(positions, rates, 0.5, color=theme.fastapi, linewidth=0)
    label_bars(axes, theme, bars, [rate_text(rate) for rate in rates])
    names = zip(
        panel.column("topology"), panel.column("reranker_capacity"), strict=True
    )
    # The capacity on two lines of its own under the topology's name.
    ticks = [f"{name}\n{wrapped(str(capacity))}" for name, capacity in names]
    axes.set_xticks(positions, ticks)
    axes.set_ylim(0, max(rates) * 1.15)  # room for the value above the highest bar
