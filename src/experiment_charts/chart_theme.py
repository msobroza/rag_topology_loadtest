"""The two themes of the charts, and the helpers that apply one to a figure.

Every chart is drawn twice, light and dark, with the same marks: only the colours
change. Blue always means the FastAPI model server and orange OVMS. Figures are
built with ``matplotlib.figure.Figure`` (never ``pyplot``), so nothing needs a
display.

Example:
    >>> figure, (axes,) = themed_figure(light_theme(), panels=1)
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.typing import LegendLocType


@dataclass(frozen=True)
class ChartTheme:
    """The colour roles of one theme and the sizes of the marks.

    ``steps`` is the ordered blue for one server and an ordered second dimension
    (small to large machine, few to many users): the last step is always the one
    farthest from the surface, so "more" stands out in both themes.
    """

    surface: str
    ink: str
    ink_secondary: str
    grid: str
    reference: str  # context lines and de-emphasised marks
    fastapi: str
    ovms: str
    steps: tuple[str, str, str]
    line_width: float = 1.5  # points: 2 px at the 100 dpi base
    marker_size: float = 7.0  # points: about 10 px across
    marker_ring: float = 1.5  # points: the 2 px ring in the surface colour
    title_size: float = 11.0
    text_size: float = 10.0
    panel_title_width: int = 56  # characters a line, so a title fits its panel

    def server_colour(self, server: str) -> str:
        """The hue of a model server, the same in every chart.

        Args:
            server: ``"FastAPI"`` or ``"OVMS"``.

        Returns:
            Its colour.

        Raises:
            ValueError: The server is neither.

        Example:
            >>> light_theme().server_colour("OVMS")
            '#eb6834'
        """
        colours = {"FastAPI": self.fastapi, "OVMS": self.ovms}
        if server not in colours:
            raise ValueError(f"server {server!r} is not one of {sorted(colours)}")
        return colours[server]

    def ordered_steps(self, count: int) -> tuple[str, ...]:
        """The blue steps for an ordered dimension of two or three levels.

        Args:
            count: How many levels; two take the first and the last step.

        Returns:
            One colour per level, smallest first.

        Raises:
            ValueError: ``count`` is not 2 or 3.

        Example:
            >>> light_theme().ordered_steps(2)
            ('#86b6ef', '#104281')
        """
        if count not in (2, 3):
            raise ValueError(f"an ordered dimension has 2 or 3 levels, not {count}")
        return (
            self.steps if count == len(self.steps) else (self.steps[0], self.steps[-1])
        )


def light_theme() -> ChartTheme:
    """The light theme.

    Returns:
        Its colour roles.

    Example:
        >>> light_theme().surface
        '#fcfcfb'
    """
    return ChartTheme(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink_secondary="#52514e",
        grid="#e6e5e0",
        reference="#8a8984",
        fastapi="#2a78d6",
        ovms="#eb6834",
        steps=("#86b6ef", "#2a78d6", "#104281"),
    )


def dark_theme() -> ChartTheme:
    """The dark theme; its blue steps run light to dark reversed on purpose.

    Returns:
        Its colour roles.

    Example:
        >>> dark_theme().surface
        '#1a1a19'
    """
    return ChartTheme(
        surface="#1a1a19",
        ink="#f0efec",
        ink_secondary="#c3c2b7",
        grid="#383835",
        reference="#8a8984",
        fastapi="#3987e5",
        ovms="#d95926",
        steps=("#1c5cab", "#5598e7", "#b7d3f6"),
    )


def themed_figure(
    theme: ChartTheme, panels: int, *, share_x: bool = False, share_y: bool = False
) -> tuple[Figure, list[Axes]]:
    """A figure of ``panels`` panels side by side, themed.

    Args:
        theme: The theme to apply.
        panels: One or two panels: 8 by 4.5 inches for one, 10 by 4.5 for two.
        share_x: Whether the panels share the x axis.
        share_y: Whether the panels share the y axis.

    Returns:
        The figure and its axes, left to right.

    Example:
        >>> figure, (left, right) = themed_figure(dark_theme(), panels=2)
    """
    width = 8.0 if panels == 1 else 10.0
    figure = Figure(figsize=(width, 4.5), dpi=100, layout="constrained")
    figure.set_facecolor(theme.surface)
    grid = figure.subplots(1, panels, sharex=share_x, sharey=share_y, squeeze=False)
    axes = list(grid[0])
    for each in axes:
        style_axes(each, theme)
    return figure, axes


def style_axes(axes: Axes, theme: ChartTheme) -> None:
    """Theme one panel: surface, no top or right spine, a faint horizontal grid.

    Args:
        axes: The panel.
        theme: The theme to apply.

    Example:
        >>> style_axes(axes, light_theme())
    """
    axes.set_facecolor(theme.surface)
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["left", "bottom"]].set_color(theme.reference)
    axes.tick_params(colors=theme.ink_secondary, labelsize=theme.text_size)
    axes.tick_params(axis="x", length=0)
    axes.yaxis.grid(True, color=theme.grid, linestyle="-", linewidth=0.8)
    axes.xaxis.grid(False)
    axes.set_axisbelow(True)


def figure_title(figure: Figure, theme: ChartTheme, title: str) -> None:
    """Put the chart's title, a plain sentence, at the top left.

    Args:
        figure: The chart's figure.
        theme: The theme to apply.
        title: What the chart plots.

    Example:
        >>> figure_title(figure, light_theme(), "One reranker under more users")
    """
    figure.suptitle(
        title, x=0.01, ha="left", color=theme.ink, fontsize=theme.title_size
    )


def panel_title(axes: Axes, theme: ChartTheme, title: str) -> None:
    """Name one panel of a two-panel chart, on two lines if it is long.

    Args:
        axes: The panel.
        theme: The theme to apply.
        title: What the panel plots.

    Example:
        >>> panel_title(axes, light_theme(), "reranker on 8 vCPUs")
    """
    # Two lines of about equal length, rather than a long line and an orphan word.
    fits = len(title) <= theme.panel_title_width
    lines = title if fits else textwrap.fill(title, math.ceil(len(title) / 2) + 6)
    axes.set_title(lines, loc="left", color=theme.ink, fontsize=theme.text_size)


def axis_labels(axes: Axes, theme: ChartTheme, x_label: str, y_label: str) -> None:
    """Label both axes in plain words with their unit, in the secondary ink.

    Args:
        axes: The panel.
        theme: The theme to apply.
        x_label: The x axis's label, or ``""`` for none.
        y_label: The y axis's label, or ``""`` for none.

    Example:
        >>> axis_labels(axes, light_theme(), "users at once", "requests a second")
    """
    size = theme.text_size
    axes.set_xlabel(x_label, color=theme.ink_secondary, fontsize=size)
    axes.set_ylabel(y_label, color=theme.ink_secondary, fontsize=size)


def add_legend(axes: Axes, theme: ChartTheme, location: LegendLocType = "best") -> None:
    """A frameless legend of the panel's labelled marks, its text in ink.

    Args:
        axes: The panel.
        theme: The theme to apply.
        location: Where matplotlib puts it, e.g. ``"upper left"``.

    Example:
        >>> add_legend(axes, light_theme(), "upper left")
    """
    axes.legend(
        loc=location, frameon=False, labelcolor=theme.ink, fontsize=theme.text_size
    )
