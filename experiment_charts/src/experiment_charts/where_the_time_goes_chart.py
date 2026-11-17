"""Chart 1: the median time of each stage of ``/retrieve``, one panel per run.

Example:
    >>> WhereTheTimeGoesChart(panels, users=4).table(registry).column("median_s")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base_experiment_chart import BaseExperimentChart
from .chart_marks import seconds_text
from .chart_selections import LabelledRun
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
class WhereTheTimeGoesChart(BaseExperimentChart):
    """Horizontal bars of the three stages' median times, one panel per run."""

    panels: tuple[LabelledRun, ...]  # each panel's title and end-to-end experiment
    users: int
    slug: ClassVar[str] = "01-where-the-time-goes"
    title: ClassVar[str] = (
        "Median time spent in each stage of a request, 4 users at once"
    )
    # The STAGE rows of /retrieve, in the order a request runs them.
    stages: ClassVar[tuple[tuple[str, str], ...]] = (
        ("/retrieve:embed", "embed the question"),
        ("/retrieve:retrieve", "search the index"),
        ("/retrieve:rerank", "rerank the 20 passages"),
    )

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """The median of each stage, in seconds, per panel.

        Args:
            registry: The experiment registry.

        Returns:
            One row per bar: panel, experiment, users, stage, label, median_s.

        Example:
            >>> chart.table(registry).column("stage")[:3]
            ('/retrieve:embed', '/retrieve:retrieve', '/retrieve:rerank')
        """
        rows = []
        for panel in self.panels:
            medians = _stage_medians_ms(registry, panel.experiment, self.users)
            rows += [
                (
                    panel.label,
                    panel.experiment,
                    self.users,
                    name,
                    label,
                    medians[name] / 1000,
                )
                for name, label in self.stages
            ]
        columns = ("panel", "experiment", "users", "stage", "label", "median_s")
        return ChartTable(columns, tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Two panels of three horizontal bars, each bar's value printed at its end.

        Args:
            table: What ``table`` returned.
            theme: The light or the dark theme.

        Returns:
            The figure.

        Example:
            >>> chart.draw(chart.table(registry), light_theme())
        """
        figure, axes = themed_figure(theme, panels=2, share_x=True, share_y=True)
        figure_title(figure, theme, self.title)
        longest = max(table.numbers("median_s"))
        for each, panel in zip(axes, table.distinct("panel"), strict=True):
            _draw_panel(each, theme, table.rows_where("panel", panel))
            panel_title(each, theme, str(panel))
            axis_labels(each, theme, "seconds", "")
            each.set_xlim(0, longest * 1.25)  # room for the value at a bar's end
        axes[0].invert_yaxis()  # the first stage on top; the y axis is shared
        return figure


def _draw_panel(axes: Axes, theme: ChartTheme, panel: ChartTable) -> None:
    positions = range(len(panel.rows))
    seconds = panel.numbers("median_s")
    bars = axes.barh(positions, seconds, height=0.5, color=theme.fastapi, linewidth=0)
    axes.set_yticks(positions, [str(label) for label in panel.column("label")])
    axes.bar_label(
        bars,
        labels=[seconds_text(value) for value in seconds],
        padding=4,
        color=theme.ink,
        fontsize=theme.text_size,
    )
    axes.yaxis.grid(False)  # grid lines would cross the bars; the values are printed
    axes.tick_params(axis="y", length=0)


def _stage_medians_ms(
    registry: ExperimentRegistry, experiment: str, users: int
) -> dict[str, int]:
    return {row.name: row.p50_ms for row in registry.stage_rows(experiment, users)}
