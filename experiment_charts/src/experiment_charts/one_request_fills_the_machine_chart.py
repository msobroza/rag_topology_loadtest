"""Chart 7: one request already uses the whole machine.

Left, the FastAPI model server's calibration: seconds for one request by the
threads given to it. Right, OVMS's calibration under load: seconds until 16
requests sent at once are all answered, by streams, every stream sharing all the
machine's vCPUs.

Example:
    >>> OneRequestFillsTheMachineChart(thread_trials, stream_trials).table(registry)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_experiment_chart import BaseExperimentChart
from .chart_marks import label_point, line_with_markers
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
class ThreadTrials:
    """A FastAPI calibration: its machine, experiment and key of thread trials."""

    label: str
    experiment: str
    trials_key: str  # under "calibration", e.g. "trials_at_batch_2"


@dataclass(frozen=True)
class StreamTrials:
    """An OVMS calibration for 16 requests at once, on a machine of ``vcpus``."""

    label: str
    experiment: str
    vcpus: int


@dataclass(frozen=True)
class OneRequestFillsTheMachineChart(BaseExperimentChart):
    """Seconds by threads (FastAPI) and by streams (OVMS), as categories."""

    thread_trials: tuple[ThreadTrials, ...]  # smallest machine first
    stream_trials: tuple[StreamTrials, ...]  # smallest machine first
    slug: ClassVar[str] = "07-one-request-fills-the-machine"
    title: ClassVar[str] = "One request already uses the whole machine"
    threads_panel: ClassVar[str] = "Seconds for one request, by threads given to it"
    streams_panel: ClassVar[str] = (
        "Seconds until 16 requests sent at once are all answered, by OVMS streams"
    )
    streams_key: ClassVar[str] = "trials_seconds_for_16_at_once"
    columns: ClassVar[tuple[str, ...]] = (
        "panel",
        "machine",
        "experiment",
        "batch_size",
        "threads",
        "streams",
        "requests_at_once",
        "seconds",
    )

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Every trial plotted; a cell that does not apply is empty.

        Args:
            registry: The experiment registry.

        Returns:
            One row per point, sorted by threads or streams within a machine.

        Example:
            >>> chart.table(registry).column("seconds")[:4]
            (4.93, 2.51, 1.34, 1.24)
        """
        rows: list[tuple[Cell, ...]] = []
        for threads in self.thread_trials:
            rows += self._thread_rows(registry, threads)
        for streams in self.stream_trials:
            rows += self._stream_rows(registry, streams)
        return ChartTable(self.columns, tuple(rows))

    def _thread_rows(
        self, registry: ExperimentRegistry, trials: ThreadTrials
    ) -> list[tuple[Cell, ...]]:
        found = registry.entry_value(
            trials.experiment, "calibration", trials.trials_key
        )
        chosen = registry.entry_value(trials.experiment, "calibration", "chosen")
        batch = parse_trial_key(trials.trials_key.removeprefix("trials_at_"))["batch"]
        by_threads = {
            parse_trial_key(key)["threads"]: (batch, seconds)
            for key, seconds in _numbers_by_key(found).items()
        }
        picked = _numbers_by_key(chosen)
        by_threads[int(picked["threads"])] = (
            int(picked["batch_size"]),
            picked["seconds_per_request"],
        )
        head = (self.threads_panel, trials.label, trials.experiment)
        return [
            (*head, size, threads, "", 1, seconds)
            for threads, (size, seconds) in sorted(by_threads.items())
        ]

    def _stream_rows(
        self, registry: ExperimentRegistry, trials: StreamTrials
    ) -> list[tuple[Cell, ...]]:
        found = _numbers_by_key(
            _trials_in_entry(registry, trials.experiment, self.streams_key)
        )
        parsed = [(parse_trial_key(key), seconds) for key, seconds in found.items()]
        on_all = sorted(
            (counts["streams"], seconds)
            for counts, seconds in parsed
            if counts["threads"] == trials.vcpus
        )
        head = (self.streams_panel, trials.label, trials.experiment, "")
        return [
            (*head, trials.vcpus, streams, 16, seconds) for streams, seconds in on_all
        ]

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Threads on the left in blue steps, streams on the right in OVMS orange.

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
        threads = table.rows_where("panel", self.threads_panel)
        machines = threads.distinct("machine")
        colours = theme.ordered_steps(len(machines))
        _draw_series(left, theme, threads, "threads", colours)
        add_legend(left, theme, "upper right")
        streams = table.rows_where("panel", self.streams_panel)
        orange = (theme.ovms,) * len(streams.distinct("machine"))
        _draw_series(right, theme, streams, "streams", orange)
        add_legend(right, theme, "lower right")
        panel_title(left, theme, self.threads_panel)
        panel_title(right, theme, self.streams_panel)
        axis_labels(left, theme, "threads", "seconds")
        axis_labels(right, theme, "OVMS streams", "seconds")
        return figure


def _draw_series(
    axes: Axes,
    theme: ChartTheme,
    panel: ChartTable,
    column: str,
    colours: tuple[str, ...],
) -> None:
    categories = sorted({int(value) for value in panel.numbers(column)})
    markers = ("o", "s")  # tells two series of one colour apart
    for index, machine in enumerate(panel.distinct("machine")):
        rows = panel.rows_where("machine", machine)
        positions = [categories.index(int(value)) for value in rows.numbers(column)]
        xy = (positions, rows.numbers("seconds"))
        line_with_markers(axes, theme, xy, colours[index], str(machine), markers[index])
        label_point(axes, theme, (xy[0][-1], xy[1][-1]), str(machine))
    axes.set_xticks(range(len(categories)), [str(count) for count in categories])
    axes.set_xlim(-0.4, len(categories) + 0.6)  # room for a direct label at the end
    axes.set_ylim(bottom=0)


def _trials_in_entry(registry: ExperimentRegistry, experiment: str, key: str) -> object:
    # A calibration run on its own records its trials under "calibration", one
    # that only measured under "measurements".
    entry = registry.entry(experiment)
    for section in ("calibration", "measurements"):
        found = entry.get(section)
        if isinstance(found, Mapping) and key in found:
            return found[key]
    raise LookupError(f"{experiment} has no {key} under calibration or measurements")


def _numbers_by_key(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping of trials, found {value!r}")
    return {str(key): number(seconds) for key, seconds in value.items()}
