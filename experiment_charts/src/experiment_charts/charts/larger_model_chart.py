"""Chart 11: a reranker five times larger, one reranker alone at 4 users.

Left, the FastAPI model server with the small and the large reranker on 8 and 16
vCPUs. Right, the large reranker on both model servers, with uniform and with
mixed-length passages.

Example:
    >>> LargerModelChart(size_pairs, size_labels, server_pairs, users=4).table(registry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.figure import Figure

from ..base_experiment_chart import BaseExperimentChart
from ..chart_marks import label_bars, rate_text, server_colours, table_bars
from ..chart_selections import RunPair, server_pair_rows
from ..chart_table import Cell, ChartTable
from ..chart_theme import (
    ChartTheme,
    add_legend,
    axis_labels,
    figure_title,
    panel_title,
    themed_figure,
)
from ..registry_reading import ExperimentRegistry


@dataclass(frozen=True)
class LargerModelChart(BaseExperimentChart):
    """Grouped bars: model size by machine (left), server by passages (right)."""

    size_pairs: tuple[RunPair, ...]  # a machine, the small and the large reranker
    size_labels: tuple[str, str]  # the small reranker's name, the large one's
    server_pairs: tuple[RunPair, ...]
    users: int
    slug: ClassVar[str] = "11-larger-model"
    title: ClassVar[str] = (
        "A reranker five times larger (4 users at once, one reranker)"
    )
    sizes_panel: ClassVar[str] = "Requests a second, FastAPI model server"
    servers_panel: ClassVar[str] = "The large reranker on both servers"
    columns: ClassVar[tuple[str, ...]] = (
        "panel",
        "group",
        "series",
        "experiment",
        "users",
        "requests_per_s",
    )

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Each bar's requests a second; ``series`` is the model or the server.

        Args:
            registry: The experiment registry.

        Returns:
            One row per bar: panel, group, series, experiment, users,
            requests_per_s.

        Example:
            >>> chart.table(registry).column("requests_per_s")[:2]
            (1.004, 0.166)
        """
        rows: list[tuple[Cell, ...]] = []
        for pair in self.size_pairs:
            for label, experiment in zip(
                self.size_labels, (pair.first, pair.second), strict=True
            ):
                rate = registry.route_row(experiment, self.users).requests_per_s
                rows.append(
                    (self.sizes_panel, pair.label, label, experiment, self.users, rate)
                )
        pair_rows = server_pair_rows(registry, self.server_pairs, self.users)
        rows += [(self.servers_panel, *pair_row) for pair_row in pair_rows]
        return ChartTable(self.columns, tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Model sizes on the left with their values, servers on the right.

        Args:
            table: What ``table`` returned.
            theme: The light or the dark theme.

        Returns:
            The figure.

        Example:
            >>> chart.draw(chart.table(registry), light_theme())
        """
        figure, (left, right) = themed_figure(theme, panels=2)
        figure_title(figure, theme, self.title)
        columns = ("group", "series", "requests_per_s")
        sizes = table.rows_where("panel", self.sizes_panel)
        steps = theme.ordered_steps(len(self.size_labels))
        for bars in table_bars(
            left, theme, sizes, columns, dict(zip(self.size_labels, steps, strict=True))
        ):
            label_bars(left, theme, bars, [rate_text(bar.get_height()) for bar in bars])
        left.set_ylim(
            0, max(sizes.numbers("requests_per_s")) * 1.4
        )  # room for the legend
        servers = table.rows_where("panel", self.servers_panel)
        table_bars(right, theme, servers, columns, server_colours(theme))
        for each, title in ((left, self.sizes_panel), (right, self.servers_panel)):
            panel_title(each, theme, title)
            axis_labels(each, theme, "", "requests a second")
            add_legend(each, theme, "upper left")
        return figure
