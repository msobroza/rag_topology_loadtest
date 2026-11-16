"""Marks several charts draw: grouped bars, lines with markers, value labels.

Each takes the theme, so a mark looks the same in every chart: lines 2 px wide,
markers ringed in the surface colour, thin bars with a gap, text in ink.

Example:
    >>> columns = ("group", "server", "requests_per_s")
    >>> bars = table_bars(axes, theme, table, columns, server_colours(theme))
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence

from matplotlib.axes import Axes
from matplotlib.container import BarContainer

from .chart_selections import SERVERS
from .chart_table import ChartTable
from .chart_theme import ChartTheme


def table_bars(
    axes: Axes,
    theme: ChartTheme,
    table: ChartTable,
    columns: tuple[str, str, str],
    colours: Mapping[str, str],
) -> list[BarContainer]:
    """Draw grouped bars from a table: one group per value of a column.

    Args:
        axes: The panel.
        theme: The theme to apply.
        table: The rows to draw, one per bar.
        columns: The group column, the series column and the value column.
        colours: Each series' colour, by its name.

    Returns:
        One container of bars per series, in order of appearance.

    Example:
        >>> columns = ("users", "server", "failed_percent")
        >>> table_bars(axes, theme, table, columns, server_colours(theme))
    """
    group_column, series_column, value_column = columns
    groups = table.distinct(group_column)
    names = [str(name) for name in table.distinct(series_column)]
    width = 0.72 / len(names)
    drawn = []
    for index, name in enumerate(names):
        rows = table.rows_where(series_column, name)
        offset = (index - (len(names) - 1) / 2) * width
        positions = [
            groups.index(group) + offset for group in rows.column(group_column)
        ]
        bars = axes.bar(
            positions,
            rows.numbers(value_column),
            width * 0.8,  # the gap between neighbouring bars
            color=colours[name],
            linewidth=0,
            label=name if len(names) > 1 else None,
        )
        drawn.append(bars)
    axes.set_xticks(range(len(groups)), [wrapped(str(group)) for group in groups])
    axes.tick_params(axis="x", colors=theme.ink_secondary)
    return drawn


def label_bars(
    axes: Axes, theme: ChartTheme, bars: BarContainer, texts: Sequence[str]
) -> None:
    """Print a text above each bar, in ink.

    Args:
        axes: The panel holding the bars.
        theme: The theme to apply.
        bars: The bars, as ``table_bars`` returns them.
        texts: One text per bar.

    Example:
        >>> label_bars(axes, theme, bars, ["0.65", "0.68"])
    """
    axes.bar_label(
        bars, labels=list(texts), padding=3, color=theme.ink, fontsize=theme.text_size
    )


def line_with_markers(
    axes: Axes,
    theme: ChartTheme,
    points: tuple[Sequence[float], Sequence[float]],
    colour: str,
    label: str | None,
    marker: str = "o",
) -> None:
    """Draw a line through ``points``, a ringed marker on each.

    Args:
        axes: The panel.
        theme: The theme to apply.
        points: The x values and the y values.
        colour: The series' colour.
        label: Its legend entry, or ``None`` for a single series.
        marker: The marker's shape, e.g. ``"o"`` or ``"s"``.

    Example:
        >>> line_with_markers(axes, theme, ((2, 8), (0.28, 1.0)), "#86b6ef", None)
    """
    axes.plot(
        *points,
        color=colour,
        linewidth=theme.line_width,
        marker=marker,
        markersize=theme.marker_size,
        markeredgecolor=theme.surface,
        markeredgewidth=theme.marker_ring,
        label=label,
    )


def label_point(
    axes: Axes,
    theme: ChartTheme,
    point: tuple[float, float],
    text: str,
    place: str = "right",
) -> None:
    """Print a text next to a point, in ink (a value or a direct label).

    Args:
        axes: The panel.
        theme: The theme to apply.
        point: The point's x and y.
        text: What to print.
        place: ``"right"``, ``"above"`` or ``"below"`` the point.

    Example:
        >>> label_point(axes, theme, (16, 1.808), "1.81", "above")
    """
    offsets = {
        "right": ((8, 0), "left", "center"),
        "above": ((0, 9), "center", "bottom"),
        "below": ((0, -9), "center", "top"),
    }
    offset, across, along = offsets[place]
    axes.annotate(
        text,
        point,
        xytext=offset,
        textcoords="offset points",
        ha=across,
        va=along,
        color=theme.ink,
        fontsize=theme.text_size,
    )


def wrapped(label: str) -> str:
    r"""A category's label on short lines, to fit under its bars.

    Args:
        label: The label, e.g. ``"16 vCPUs, mixed passages"``.

    Returns:
        The label broken into lines of at most 12 characters where it can be.

    Example:
        >>> wrapped("16 vCPUs, mixed passages")
        '16 vCPUs,\nmixed\npassages'
    """
    return textwrap.fill(label, 12)


def seconds_text(seconds: float) -> str:
    """A time in seconds as printed on a chart, at least one decimal.

    The registry's times are whole milliseconds, so three decimals say all of
    them; trailing zeros go, except the one that keeps "1.0 s" from reading "1 s".

    Args:
        seconds: The time.

    Returns:
        The text, e.g. ``"0.015 s"``, ``"3.4 s"`` or ``"1.0 s"``.

    Example:
        >>> seconds_text(0.015)
        '0.015 s'
    """
    digits = f"{seconds:.3f}".rstrip("0")
    return f"{digits}0 s" if digits.endswith(".") else f"{digits} s"


def rate_text(requests_per_s: float) -> str:
    """Requests a second as printed on a chart: two decimals.

    Args:
        requests_per_s: The throughput.

    Returns:
        The text, e.g. ``"1.81"``.

    Example:
        >>> rate_text(1.808)
        '1.81'
    """
    return f"{requests_per_s:.2f}"


def server_colours(theme: ChartTheme) -> dict[str, str]:
    """Each model server's hue, by the name a table gives it.

    Args:
        theme: The light or the dark theme.

    Returns:
        ``"FastAPI"`` and ``"OVMS"`` with their colours.

    Example:
        >>> server_colours(light_theme())["OVMS"]
        '#eb6834'
    """
    return {server: theme.server_colour(server) for server in SERVERS}
