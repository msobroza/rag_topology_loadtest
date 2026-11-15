"""Chart 2: one reranker under more and more users, throughput and wait.

The right panel adds, for each machine, the wait the throughput alone predicts:
with ``users`` requests always in flight and ``requests_per_s`` answered a second,
each waits about ``users / requests_per_s`` seconds (Little's law).

Example:
    >>> MoreUsersMoreWaitingChart(machines, users=(1, 4, 8, 16, 32)).table(registry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_experiment_chart import BaseExperimentChart
from .chart_marks import label_point, line_with_markers
from .chart_selections import LabelledRun
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
class MoreUsersMoreWaitingChart(BaseExperimentChart):
    """Requests a second and median wait against users at once, per machine."""

    machines: tuple[LabelledRun, ...]  # smallest machine first
    users: tuple[int, ...]
    slug: ClassVar[str] = "02-more-users-more-waiting"
    title: ClassVar[str] = "One reranker under more and more users"

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Each run's requests a second, median wait and predicted wait.

        Args:
            registry: The experiment registry.

        Returns:
            One row per machine and user count: machine, experiment, users,
            requests_per_s, median_s, expected_wait_s.

        Example:
            >>> chart.table(registry).column("users")[:2]
            (1, 4)
        """
        rows = []
        for machine in self.machines:
            for users in self.users:
                route = registry.route_row(machine.experiment, users)
                expected = users / route.requests_per_s
                median = route.p50_ms / 1000
                rows.append(
                    (
                        machine.label,
                        machine.experiment,
                        users,
                        route.requests_per_s,
                        median,
                        expected,
                    )
                )
        columns = ("machine", "experiment", "users", "requests_per_s", "median_s")
        return ChartTable((*columns, "expected_wait_s"), tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Requests a second on the left, waits on the right, users on a linear axis.

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
        machines = table.distinct("machine")
        colours = theme.ordered_steps(len(machines))
        for index, (colour, machine) in enumerate(zip(colours, machines, strict=True)):
            runs = table.rows_where("machine", machine)
            # One legend entry stands for every machine's predicted-wait line.
            reference = (
                "users ÷ requests a second" if index == len(machines) - 1 else None
            )
            _draw_machine((left, right), theme, runs, colour, reference)
        _finish_panels(left, right, theme, self.users)
        return figure


def _draw_machine(
    panels: tuple[Axes, Axes],
    theme: ChartTheme,
    runs: ChartTable,
    colour: str,
    reference: str | None,
) -> None:
    left, right = panels
    positions = runs.numbers("users")
    label = str(runs.column("machine")[0])
    line_with_markers(
        left, theme, (positions, runs.numbers("requests_per_s")), colour, label
    )
    line_with_markers(
        right, theme, (positions, runs.numbers("median_s")), colour, label
    )
    right.plot(
        positions,
        runs.numbers("expected_wait_s"),
        color=theme.reference,
        linewidth=1.0,
        label=reference,
        zorder=1,  # under the measured line
    )
    end = (positions[-1], runs.numbers("median_s")[-1])
    label_point(right, theme, end, label)


def _finish_panels(
    left: Axes, right: Axes, theme: ChartTheme, users: tuple[int, ...]
) -> None:
    # Linear in users, so a wait that grows with the queue draws as a straight line;
    # the room on the right holds the labels at the ends of the lines.
    left.set_xticks(users, [str(count) for count in users])
    left.set_xlim(0, users[-1] * 1.25)
    left.set_ylim(bottom=0)
    right.set_ylim(bottom=0)
    panel_title(left, theme, "Requests answered a second")
    panel_title(right, theme, "Median wait for an answer")
    axis_labels(left, theme, "users at once", "requests a second")
    axis_labels(right, theme, "users at once", "seconds")
    add_legend(left, theme, "lower right")
    add_legend(right, theme, "upper left")
