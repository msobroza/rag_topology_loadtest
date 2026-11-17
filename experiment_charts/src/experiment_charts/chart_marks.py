"""Marks several charts draw: grouped bars, lines with markers, value labels.

And the two panels of a series against users at once (charts 2 and 4).

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
from matplotlib.typing import LegendLocType

from .chart_selections import SERVERS
from .chart_table import ChartTable
from .chart_theme import ChartTheme, add_legend, axis_labels, panel_title


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


def generator_server_colours(theme: ChartTheme) -> dict[str, str]:
    """Each generator server's hue: OVMS's orange, PyTorch's blue, vLLM neutral.

    PyTorch takes the FastAPI model server's blue because it is served behind
    FastAPI; vLLM has no colour of its own in the theme, so it takes the neutral
    reference rather than a hue that would mean another server elsewhere.

    Args:
        theme: The light or the dark theme.

    Returns:
        ``"OVMS"``, ``"PyTorch"`` and ``"vLLM"`` with their colours.

    Example:
        >>> generator_server_colours(light_theme())["vLLM"]
        '#8a8984'
    """
    return {"OVMS": theme.ovms, "PyTorch": theme.fastapi, "vLLM": theme.reference}


def draw_users_series(
    panels: tuple[Axes, Axes],
    theme: ChartTheme,
    runs: ChartTable,
    colour: str,
    label: str,
) -> None:
    """Draw one series against users at once: throughput left, median wait right.

    The right panel's line gets its label at its end, so the reader need not look
    up the legend there.

    Args:
        panels: The left and the right panel.
        theme: The theme to apply.
        runs: The series' rows, with ``users``, ``requests_per_s`` and ``median_s``.
        colour: The series' colour.
        label: Its name, in the legend and at the end of the right line.

    Example:
        >>> draw_users_series((left, right), theme, runs, theme.steps[0], "8 vCPUs")
    """
    left, right = panels
    users = runs.numbers("users")
    rates, medians = runs.numbers("requests_per_s"), runs.numbers("median_s")
    line_with_markers(left, theme, (users, rates), colour, label)
    line_with_markers(right, theme, (users, medians), colour, label)
    label_point(right, theme, (users[-1], medians[-1]), label)


def finish_users_panels(
    panels: tuple[Axes, Axes],
    theme: ChartTheme,
    users: tuple[int, ...],
    legend_places: tuple[LegendLocType, LegendLocType],
    end_labels: tuple[str, ...],
) -> None:
    """Title, label and scale the two panels ``draw_users_series`` drew on.

    Args:
        panels: The left and the right panel, sharing their x axis.
        theme: The theme to apply.
        users: The user counts, ticks of the linear x axis.
        legend_places: Where each panel's legend goes, clear of its lines.
        end_labels: The labels drawn at the ends of the right panel's lines.

    Example:
        >>> places = ("lower right", "upper left")
        >>> finish_users_panels((left, right), theme, (1, 4, 16), places, ("8 vCPUs",))
    """
    left, right = panels
    # Linear in users, so a wait that grows with the queue draws as a straight line.
    # The room on the right holds the labels at the ends of the lines: it grows with
    # the longest of them, or a long label would run off the figure.
    longest = max(len(label) for label in end_labels)
    left.set_xticks(users, [str(count) for count in users])
    left.set_xlim(0, users[-1] * (1.1 + 0.018 * longest))
    for each, title, y_label, place in (
        (left, "Requests answered a second", "requests a second", legend_places[0]),
        (right, "Median wait for an answer", "seconds", legend_places[1]),
    ):
        each.set_ylim(bottom=0)
        panel_title(each, theme, title)
        axis_labels(each, theme, "users at once", y_label)
        add_legend(each, theme, place)
