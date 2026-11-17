"""Chart 8: small batches score mixed-length passages faster, and what it gives.

Left, the FastAPI model server's calibration on three machines: the time of one
request by batch size, as a percentage of one batch of 20 on the same machine.
Right, both servers through the whole workflow, whose passages are of mixed length.

Example:
    >>> SmallBatchesChart(trials, (20, 10, 5, 2, 1), pairs, users=4).table(registry)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_experiment_chart import BaseExperimentChart
from .chart_marks import (
    label_bars,
    line_with_markers,
    rate_text,
    server_colours,
    table_bars,
)
from .chart_selections import RunPair, server_pair_rows
from .chart_table import Cell, ChartTable
from .chart_theme import (
    ChartTheme,
    add_legend,
    axis_labels,
    figure_title,
    panel_title,
    themed_figure,
)
from .registry_reading import ExperimentRegistry, number, parse_trial_key


@dataclass(frozen=True)
class BatchTrials:
    """One machine's calibration trials by batch size, where the entry holds them.

    ``keys`` lead from the entry's ``calibration`` to the trials: either a mapping
    ``batch_<n>: seconds``, or a list of ``{threads, batch_size, seconds}`` of
    which the ones on ``threads`` threads are read.
    """

    label: str
    experiment: str
    keys: tuple[str, ...]
    threads: int


@dataclass(frozen=True)
class SmallBatchesChart(BaseExperimentChart):
    """Relative time by batch size (left); end-to-end requests a second (right)."""

    batch_trials: tuple[BatchTrials, ...]  # smallest machine first
    batch_sizes: tuple[int, ...]  # the first is the reference, 100 %
    workflow_pairs: tuple[RunPair, ...]
    users: int
    slug: ClassVar[str] = "08-small-batches"
    title: ClassVar[str] = (
        "Scoring 20 passages of mixed length: "
        "time by batch size, relative to one batch of 20"
    )
    batches_panel: ClassVar[str] = "Time of one request, by batch size"
    workflow_panel: ClassVar[str] = (
        "Requests a second through the whole workflow, 4 users at once"
    )
    columns: ClassVar[tuple[str, ...]] = (
        "panel",
        "group",
        "server",
        "experiment",
        "batch_size",
        "seconds",
        "percent_of_batch_20",
        "users",
        "requests_per_s",
    )

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """The trials with their percentage, then the end-to-end runs.

        Args:
            registry: The experiment registry.

        Returns:
            One row per point or bar; a cell that does not apply is empty.

        Example:
            >>> chart.table(registry).column("percent_of_batch_20")[0]
            100.0
        """
        rows: list[tuple[Cell, ...]] = []
        for trials in self.batch_trials:
            seconds = _seconds_by_batch(registry, trials)
            reference = seconds[self.batch_sizes[0]]
            rows += [
                (
                    self.batches_panel,
                    trials.label,
                    "FastAPI",
                    trials.experiment,
                    size,
                    seconds[size],
                    100 * seconds[size] / reference,
                    "",
                    "",
                )
                for size in self.batch_sizes
            ]
        for pair_row in server_pair_rows(registry, self.workflow_pairs, self.users):
            group, server, experiment, users, rate = pair_row
            head = (self.workflow_panel, group, server, experiment)
            rows.append((*head, "", "", "", users, rate))
        return ChartTable(self.columns, tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Lines of percentages on the left, four labelled bars on the right.

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
        _draw_batches(left, theme, table.rows_where("panel", self.batches_panel))
        workflow = table.rows_where("panel", self.workflow_panel)
        columns = ("group", "server", "requests_per_s")
        for bars in table_bars(right, theme, workflow, columns, server_colours(theme)):
            label_bars(
                right, theme, bars, [rate_text(bar.get_height()) for bar in bars]
            )
        panel_title(left, theme, self.batches_panel)
        panel_title(right, theme, self.workflow_panel)
        axis_labels(left, theme, "passages per batch", "time, % of one batch of 20")
        axis_labels(right, theme, "", "requests a second")
        add_legend(left, theme, "lower left")
        add_legend(right, theme, "upper left")
        return figure


def _draw_batches(axes: Axes, theme: ChartTheme, panel: ChartTable) -> None:
    machines = panel.distinct("group")
    for colour, machine in zip(
        theme.ordered_steps(len(machines)), machines, strict=True
    ):
        rows = panel.rows_where("group", machine)
        positions = list(range(len(rows.rows)))
        xy = (positions, rows.numbers("percent_of_batch_20"))
        line_with_markers(axes, theme, xy, colour, str(machine))
    sizes = panel.rows_where("group", machines[0]).column("batch_size")
    axes.set_xticks(range(len(sizes)), [str(size) for size in sizes])
    axes.set_ylim(0, 110)


def _seconds_by_batch(
    registry: ExperimentRegistry, trials: BatchTrials
) -> dict[int, float]:
    found = registry.entry_value(trials.experiment, "calibration", *trials.keys)
    if isinstance(found, Mapping):
        return {
            parse_trial_key(str(key))["batch"]: number(value)
            for key, value in found.items()
        }
    if not isinstance(found, list):
        raise TypeError(f"{trials.experiment}: expected trials, found {found!r}")
    return {
        int(trial["batch_size"]): number(trial["seconds"])
        for trial in found
        if trial["threads"] == trials.threads
    }
