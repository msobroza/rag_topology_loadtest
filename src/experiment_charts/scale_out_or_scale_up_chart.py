"""Chart 4: more machines of 2 vCPUs, or one bigger machine.

Left, requests a second through the whole workflow with default settings, for
reranker capacity spread over small machines or held by one machine of 8 vCPUs.
Right, the median time of a request when it is the only one, by the size of the
reranker's machine: a request is served on one machine, so only a bigger machine
makes it faster.

Example:
    >>> ScaleOutOrScaleUpChart(throughput, latency, approaches).table(registry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .base_experiment_chart import BaseExperimentChart
from .chart_marks import label_bars, rate_text, seconds_text
from .chart_table import Cell, ChartTable
from .chart_theme import (
    ChartTheme,
    axis_labels,
    figure_title,
    panel_title,
    themed_figure,
)
from .registry_reading import ExperimentRegistry


@dataclass(frozen=True)
class ApproachBar:
    """One bar: its label, its approach, its run and the machines running a reranker."""

    label: str  # one line, e.g. "2 machines of 2 vCPUs"
    approach: str
    experiment: str
    users: int
    machines: int
    vcpus_each: int


@dataclass(frozen=True)
class ScaleOutOrScaleUpChart(BaseExperimentChart):
    """Bars coloured by approach: throughput (left), a lone request's time (right)."""

    throughput_bars: tuple[ApproachBar, ...]
    latency_bars: tuple[ApproachBar, ...]
    approaches: tuple[str, str]  # scaling out first, then scaling up
    slug: ClassVar[str] = "04-scale-out-or-scale-up"
    title: ClassVar[str] = "More machines, or a bigger machine"
    throughput_panel: ClassVar[str] = (
        "Requests a second through the whole workflow, default settings, "
        "8 to 10 users at once"
    )
    latency_panel: ClassVar[str] = (
        "Median time of a request when it is the only one, by the reranker's machine"
    )
    columns: ClassVar[tuple[str, ...]] = (
        "panel",
        "label",
        "approach",
        "experiment",
        "users",
        "machines",
        "vcpus_each",
        "reranker_vcpus_total",
        "requests_per_s",
        "requests_per_s_per_reranker_vcpu",
        "median_s",
    )

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Each bar's route row; a cell that does not apply is empty.

        Args:
            registry: The experiment registry.

        Returns:
            One row per bar, the throughput panel's first.

        Example:
            >>> chart.table(registry).column("requests_per_s")[:3]
            (0.336, 0.649, 0.652)
        """
        rows: list[tuple[Cell, ...]] = []
        for bar in self.throughput_bars:
            rate = registry.route_row(bar.experiment, bar.users).requests_per_s
            total = bar.machines * bar.vcpus_each
            rows.append((*_head(self.throughput_panel, bar), rate, rate / total, ""))
        for bar in self.latency_bars:
            median = registry.route_row(bar.experiment, bar.users).p50_ms / 1000
            rows.append((*_head(self.latency_panel, bar), "", "", median))
        return ChartTable(self.columns, tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Two panels of three bars, every value printed, one legend for both.

        Args:
            table: What ``table`` returned.
            theme: The light or the dark theme.

        Returns:
            The figure.

        Example:
            >>> chart.draw(chart.table(registry), dark_theme())
        """
        figure, (left, right) = themed_figure(theme, panels=2)
        figure_title(figure, theme, self.title)
        colours = dict(zip(self.approaches, theme.ordered_steps(2), strict=True))
        throughput = table.rows_where("panel", self.throughput_panel)
        _draw_bars(left, theme, throughput, "requests_per_s", colours)
        latency = table.rows_where("panel", self.latency_panel)
        _draw_bars(right, theme, latency, "median_s", colours)
        panel_title(left, theme, self.throughput_panel)
        panel_title(right, theme, self.latency_panel)
        axis_labels(left, theme, "", "requests a second")
        axis_labels(right, theme, "", "seconds")
        figure.legend(
            handles=[
                Patch(color=colours[name], label=name) for name in self.approaches
            ],
            loc="outside lower center",
            ncols=2,
            frameon=False,
            labelcolor=theme.ink,
            fontsize=theme.text_size,
        )
        return figure


def _head(panel: str, bar: ApproachBar) -> tuple[Cell, ...]:
    run = (panel, bar.label, bar.approach, bar.experiment, bar.users)
    return (*run, bar.machines, bar.vcpus_each, bar.machines * bar.vcpus_each)


def _draw_bars(
    axes: Axes,
    theme: ChartTheme,
    panel: ChartTable,
    value_column: str,
    colours: dict[str, str],
) -> None:
    positions = range(len(panel.rows))
    values = panel.numbers(value_column)
    bar_colours = [colours[str(approach)] for approach in panel.column("approach")]
    bars = axes.bar(positions, values, 0.5, color=bar_colours, linewidth=0)
    is_rate = value_column == "requests_per_s"
    texts = [rate_text(value) if is_rate else seconds_text(value) for value in values]
    label_bars(axes, theme, bars, texts)
    axes.set_xticks(positions, _tick_labels(panel, with_total=is_rate))
    axes.set_ylim(0, max(values) * 1.15)  # room for the value above the highest bar


def _tick_labels(panel: ChartTable, with_total: bool) -> list[str]:
    # "2 machines of 2 vCPUs" reads on two lines; the throughput panel adds the
    # vCPUs in all, the measure the bars are compared on.
    labels = [str(label).replace(" of ", "\nof ") for label in panel.column("label")]
    if not with_total:
        return labels
    totals = panel.column("reranker_vcpus_total")
    return [
        f"{label}\n({total} vCPUs in all)"
        for label, total in zip(labels, totals, strict=True)
    ]
