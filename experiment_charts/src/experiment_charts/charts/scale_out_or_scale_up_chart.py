"""Chart 4: eight reranker vCPUs as four machines of 2 vCPUs or as one of 8.

One sitting measured both like for like: the same split topology, the same
FastAPI model server and settings, the same users, only the reranker's machines
differ. Left, requests a second through the whole workflow; right, the median
wait, both against users at once.

Example:
    >>> ScaleOutOrScaleUpChart(series, users, approaches).table(registry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.figure import Figure
from matplotlib.typing import LegendLocType

from ..base_experiment_chart import BaseExperimentChart
from ..chart_marks import draw_users_series, finish_users_panels
from ..chart_table import ChartTable
from ..chart_theme import ChartTheme, figure_title, themed_figure
from ..registry_reading import ExperimentRegistry


@dataclass(frozen=True)
class ApproachSeries:
    """One line: its label, its approach, its experiment and the reranker machines."""

    label: str  # e.g. "4 machines of 2 vCPUs"
    approach: str
    experiment: str
    machines: int
    vcpus_each: int


@dataclass(frozen=True)
class ScaleOutOrScaleUpChart(BaseExperimentChart):
    """Throughput and median wait against users, one line per approach."""

    series: tuple[ApproachSeries, ...]
    users: tuple[int, ...]
    approaches: tuple[str, str]  # scaling out first, then scaling up
    slug: ClassVar[str] = "04-scale-out-or-scale-up"
    title: ClassVar[str] = "Eight reranker vCPUs as four machines or as one"
    columns: ClassVar[tuple[str, ...]] = (
        "label",
        "approach",
        "experiment",
        "users",
        "machines",
        "vcpus_each",
        "reranker_vcpus_total",
        "requests_per_s",
        "median_s",
        "p95_s",
    )

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Each series' route row at each user count.

        Args:
            registry: The experiment registry.

        Returns:
            One row per series and user count, the scale-out series first.

        Example:
            >>> chart.table(registry).column("requests_per_s")[:2]
            (0.18, 0.377)
        """
        rows = []
        for line in self.series:
            total = line.machines * line.vcpus_each
            head = (line.label, line.approach, line.experiment)
            for users in self.users:
                route = registry.route_row(line.experiment, users)
                waits = (route.p50_ms / 1000, route.p95_ms / 1000)
                machines = (line.machines, line.vcpus_each, total)
                rows.append((*head, users, *machines, route.requests_per_s, *waits))
        return ChartTable(self.columns, tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Two panels sharing a linear axis of users, coloured by approach.

        Args:
            table: What ``table`` returned.
            theme: The light or the dark theme.

        Returns:
            The figure.

        Example:
            >>> chart.draw(chart.table(registry), dark_theme())
        """
        figure, (left, right) = themed_figure(theme, panels=2, share_x=True)
        figure_title(figure, theme, self.title)
        colours = dict(zip(self.approaches, theme.ordered_steps(2), strict=True))
        labels = tuple(str(label) for label in table.distinct("label"))
        for label in labels:
            runs = table.rows_where("label", label)
            colour = colours[str(runs.column("approach")[0])]
            draw_users_series((left, right), theme, runs, colour, label)
        # Each legend in the corner its panel's lines leave clear.
        places: tuple[LegendLocType, LegendLocType] = ("lower right", "upper left")
        finish_users_panels((left, right), theme, self.users, places, labels)
        return figure
