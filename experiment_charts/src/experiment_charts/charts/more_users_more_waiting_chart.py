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
from matplotlib.typing import LegendLocType

from ..base_experiment_chart import BaseExperimentChart
from ..chart_marks import draw_users_series, finish_users_panels
from ..chart_selections import LabelledRun
from ..chart_table import ChartTable
from ..chart_theme import ChartTheme, figure_title, themed_figure
from ..registry_reading import ExperimentRegistry


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
        # Each legend in the corner its panel's lines leave clear.
        places: tuple[LegendLocType, LegendLocType] = ("lower right", "upper left")
        labels = tuple(str(machine) for machine in machines)
        finish_users_panels((left, right), theme, self.users, places, labels)
        return figure


def _draw_machine(
    panels: tuple[Axes, Axes],
    theme: ChartTheme,
    runs: ChartTable,
    colour: str,
    reference: str | None,
) -> None:
    draw_users_series(panels, theme, runs, colour, str(runs.column("machine")[0]))
    panels[1].plot(
        runs.numbers("users"),
        runs.numbers("expected_wait_s"),
        color=theme.reference,
        linewidth=1.0,
        label=reference,
        zorder=1,  # under the measured line
    )
