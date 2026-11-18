"""Chart 13: the generator under Locust, at 1 and at 4 users at once.

Left, the tokens each server wrote a second; right, the median wait for an
answer. Tokens rather than requests, because the servers' answers are not the
same length.

Example:
    >>> GenerationUnderLoadChart(servers, users=(1, 4)).table(registry)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..base_experiment_chart import BaseExperimentChart
from ..chart_marks import label_bars, table_bars
from ..chart_selections import LabelledRun
from ..chart_table import Cell, ChartTable
from ..chart_theme import (
    ChartTheme,
    add_legend,
    axis_labels,
    figure_title,
    panel_title,
    themed_figure,
)
from ..registry_reading import ExperimentRegistry, number


@dataclass(frozen=True)
class GenerationUnderLoadChart(BaseExperimentChart):
    """Grouped bars, one group per server, one bar per user count."""

    servers: tuple[LabelledRun, ...]
    users: tuple[int, ...]
    slug: ClassVar[str] = "13-generation-under-load"
    title: ClassVar[str] = (
        "The generator under load: tokens served and the wait, at 1 and 4 users"
        " (16 vCPUs)"
    )
    columns: ClassVar[tuple[str, ...]] = (
        "server",
        "experiment",
        "users",
        "served_tokens_per_s",
        "p50_s",
    )

    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Each server's tokens served a second and median wait, by user count.

        Args:
            registry: The experiment registry.

        Returns:
            One row per server and user count: server, experiment, users,
            served_tokens_per_s, p50_s.

        Example:
            >>> chart.table(registry).column("served_tokens_per_s")[:2]
            (14.1, 11.6)
        """
        rows: list[tuple[Cell, ...]] = [
            (
                run.label,
                run.experiment,
                users,
                number(registry.entry_value(run.experiment, *_tokens_keys(users))),
                registry.route_row(run.experiment, users).p50_ms / 1000,
            )
            for run in self.servers
            for users in self.users
        ]
        return ChartTable(self.columns, tuple(rows))

    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Tokens served on the left, the median wait on the right, values printed.

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
        panels = (
            (left, "served_tokens_per_s", "Tokens served a second", "tokens a second"),
            (right, "p50_s", "Median wait for an answer, seconds", "seconds"),
        )
        for axes, column, title, y_label in panels:
            _draw_panel(axes, theme, table, column)
            panel_title(axes, theme, title)
            axis_labels(axes, theme, "", y_label)
            add_legend(axes, theme, "upper left")
        return figure


def _tokens_keys(users: int) -> tuple[str, ...]:
    return ("measurements", "tokens", f"users_{users}", "served_tokens_per_s")


def _value_text(column: str, value: float) -> str:
    # Tokens keep their one decimal ("10.0"); the waits are whole or one-decimal
    # seconds in the registry ("42", "7.9"), so they print as recorded.
    return f"{value:.1f}" if column == "served_tokens_per_s" else f"{value:g}"


def _users_text(users: Cell) -> str:
    return "1 user" if users == 1 else f"{users} users"


def _draw_panel(axes: Axes, theme: ChartTheme, table: ChartTable, column: str) -> None:
    # The bars' series is a text, so the user counts become their legend entries.
    series = [_users_text(users) for users in table.column("users")]
    rows = zip(table.column("server"), series, table.column(column), strict=True)
    view = ChartTable(("server", "series", column), tuple(rows))
    names = [str(name) for name in view.distinct("series")]
    colours = dict(zip(names, theme.ordered_steps(len(names)), strict=True))
    for bars in table_bars(axes, theme, view, ("server", "series", column), colours):
        texts = [_value_text(column, bar.get_height()) for bar in bars]
        label_bars(axes, theme, bars, texts)
    axes.set_ylim(0, max(view.numbers(column)) * 1.35)  # room for the legend
