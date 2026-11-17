"""Chart 12: one request to the generator, reading its prompt then writing.

Left, the seconds each server takes to read a 781-token prompt; right, a whole
request of a 922-token prompt and 128 tokens written. Each number is one request
sent straight to the server by a script, not a Locust run.

Example:
    >>> ReadingAndWritingChart(probes).table(registry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_experiment_chart import BaseExperimentChart
from .chart_marks import generator_server_colours, label_bars, seconds_text
from .chart_table import Cell, ChartTable
from .chart_theme import (
    ChartTheme,
    axis_labels,
    figure_title,
    panel_title,
    themed_figure,
)
from .registry_reading import ExperimentRegistry, number


@dataclass(frozen=True)
class GeneratorProbe:
    """One server's measurement and where its two times are in the entry."""

    label: str
    experiment: str
    reading_keys: tuple[str, ...]  # the keys down to the reading time, seconds
    request_keys: tuple[str, ...]  # the keys down to the whole request, seconds


@dataclass(frozen=True)
class ReadingAndWritingChart(BaseExperimentChart):
    """Plain bars, one per server: reading (left), a whole request (right)."""

    probes: tuple[GeneratorProbe, ...]
    slug: ClassVar[str] = "12-reading-and-writing"
    title: ClassVar[str] = (
        "One request to the generator: reading the prompt, then writing the answer"
        " (16 vCPUs)"
    )
    reading_panel: ClassVar[str] = "Reading a 781-token prompt"
    request_panel: ClassVar[str] = (
        "A whole request: a 922-token prompt, 128 tokens written"
    )

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Each server's seconds to read, then for the whole request.

        Args:
            registry: The experiment registry.

        Returns:
            One row per bar: panel, server, experiment, seconds.

        Example:
            >>> chart.table(registry).column("seconds")[:3]
            (3.94, 1.14, 7.91)
        """
        rows: list[tuple[Cell, ...]] = []
        for panel, keyed in (
            (self.reading_panel, [(each, each.reading_keys) for each in self.probes]),
            (self.request_panel, [(each, each.request_keys) for each in self.probes]),
        ):
            rows += [
                (panel, probe.label, probe.experiment, number(seconds))
                for probe, keys in keyed
                for seconds in (registry.entry_value(probe.experiment, *keys),)
            ]
        return ChartTable(("panel", "server", "experiment", "seconds"), tuple(rows))

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
            axis_labels(each, theme, "", "seconds")
        return figure


def _draw_panel(axes: Axes, theme: ChartTheme, panel: ChartTable) -> None:
    positions = range(len(panel.rows))
    seconds = panel.numbers("seconds")
    servers = [str(server) for server in panel.column("server")]
    colours = generator_server_colours(theme)
    bars = axes.bar(
        positions,
        seconds,
        0.5,
        color=[colours[server] for server in servers],
        linewidth=0,
    )
    label_bars(axes, theme, bars, [seconds_text(value) for value in seconds])
    axes.set_xticks(positions, servers)
    axes.set_ylim(0, max(seconds) * 1.15)  # room for the value above the highest bar
