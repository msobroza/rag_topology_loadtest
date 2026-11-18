"""Chart 3: requests a second of one calibrated reranker, by machine size.

The table adds the vCPU-seconds each request costs (``vcpus / requests_per_s``),
the measure that says whether a bigger machine is also a better use of its CPUs.

Example:
    >>> BiggerMachineMoreAnswersChart(series, users=4).table(registry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.figure import Figure

from ..base_experiment_chart import BaseExperimentChart
from ..chart_marks import label_point, line_with_markers, rate_text
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
class MachineRun:
    """The experiment of one machine size."""

    vcpus: int
    experiment: str


@dataclass(frozen=True)
class WeightSeries:
    """One line of the chart: a kind of weights, its experiments by machine size."""

    label: str
    runs: tuple[MachineRun, ...]  # smallest machine first


@dataclass(frozen=True)
class BiggerMachineMoreAnswersChart(BaseExperimentChart):
    """Requests a second against vCPUs, one line per kind of weights."""

    series: tuple[WeightSeries, ...]
    users: int
    slug: ClassVar[str] = "03-bigger-machine-more-answers"
    title: ClassVar[str] = (
        "Requests a second of one reranker, by machine size (4 users at once)"
    )

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Each machine's requests a second, and the vCPU-seconds of a request.

        Args:
            registry: The experiment registry.

        Returns:
            One row per point: weights, vcpus, experiment, users, requests_per_s,
            vcpu_seconds_per_request.

        Example:
            >>> chart.table(registry).column("requests_per_s")[:3]
            (0.282, 1.004, 1.808)
        """
        rows = []
        for weights in self.series:
            for run in weights.runs:
                rate = registry.route_row(run.experiment, self.users).requests_per_s
                rows.append(
                    (
                        weights.label,
                        run.vcpus,
                        run.experiment,
                        self.users,
                        rate,
                        run.vcpus / rate,
                    )
                )
        columns = ("weights", "vcpus", "experiment", "users", "requests_per_s")
        return ChartTable((*columns, "vcpu_seconds_per_request"), tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """One line per kind of weights, each point's value printed.

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
        names = table.distinct("weights")
        colours = theme.ordered_steps(len(names))
        # The first series' values go below its points, the last's above: the two
        # lines are close at 2 vCPUs.
        places = ("below", "above")
        for index, (colour, name) in enumerate(zip(colours, names, strict=True)):
            points = table.rows_where("weights", name)
            xy = (points.numbers("vcpus"), points.numbers("requests_per_s"))
            line_with_markers(axes, theme, xy, colour, str(name))
            for point in zip(*xy, strict=True):
                label_point(axes, theme, point, rate_text(point[1]), places[index])
            label_point(axes, theme, (xy[0][-1] + 0.4, xy[1][-1]), str(name))
        axes.set_xticks(sorted(set(table.numbers("vcpus"))))
        axes.set_xlim(0, 20.5)  # room for the direct labels right of 16 vCPUs
        axes.set_ylim(bottom=0)
        axis_labels(axes, theme, "vCPUs", "requests a second")
        add_legend(axes, theme, "upper left")
        return figure
