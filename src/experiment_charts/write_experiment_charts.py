"""Write every chart's figures (light and dark) and table: ``make experiment-charts``.

The files are reproducible: the SVG ids come from a fixed salt, the SVG carries no
date, text is drawn as paths, and the rows keep the catalog's order. So a figure
changes in a diff only when its numbers or its drawing do.

Example:
    >>> main(["--image-format", "png", "--out", "/tmp/chart-previews"])
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

from .base_experiment_chart import BaseExperimentChart
from .chart_catalog import experiment_charts
from .chart_table import ChartTable
from .chart_theme import ChartTheme, dark_theme, light_theme
from .registry_reading import ExperimentRegistry, read_experiment_registry


@dataclass(frozen=True)
class ChartOutput:
    """Where the charts are read from and written to, and in which format."""

    registry_dir: Path
    output_dir: Path
    image_format: str  # "svg" (committed) or "png" (previews)
    svg_hashsalt: str = "experiment-charts"  # fixed, so SVG ids are reproducible
    png_dpi: int = 200


def write_experiment_charts(options: ChartOutput) -> list[Path]:
    """Write each chart's two figures under ``figures/`` and its table under ``data/``.

    Args:
        options: The registry, the output folder and the image format.

    Returns:
        Every file written, in the catalog's order.

    Example:
        >>> options = ChartOutput(Path("experiments"), Path("out"), "svg")
        >>> write_experiment_charts(options)
    """
    registry = read_experiment_registry(options.registry_dir)
    for folder in ("figures", "data"):
        (options.output_dir / folder).mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for chart in experiment_charts():
        written += write_chart(chart, registry, options)
    return written


def write_chart(
    chart: BaseExperimentChart, registry: ExperimentRegistry, options: ChartOutput
) -> list[Path]:
    """Write one chart's table and its light and dark figures.

    Args:
        chart: The chart.
        registry: The experiment registry.
        options: The output folder and the image format.

    Returns:
        The table's path, then the light and the dark figure's.

    Example:
        >>> write_chart(experiment_charts()[7], registry, options)
    """
    table = chart.table(registry)
    table_path = options.output_dir / "data" / f"{chart.slug}.csv"
    table.write_csv(table_path)
    written = [table_path]
    for suffix, theme in (("", light_theme()), ("-dark", dark_theme())):
        name = f"{chart.slug}{suffix}.{options.image_format}"
        path = options.output_dir / "figures" / name
        save_figure(chart, table, theme, path, options)
        written.append(path)
    return written


def save_figure(
    chart: BaseExperimentChart,
    table: ChartTable,
    theme: ChartTheme,
    path: Path,
    options: ChartOutput,
) -> None:
    """Draw ``chart`` from ``table`` in ``theme`` and save it at ``path``.

    Args:
        chart: The chart.
        table: Its table.
        theme: The light or the dark theme.
        path: The file to write.
        options: The image format and its reproducibility settings.

    Example:
        >>> save_figure(chart, chart.table(registry), dark_theme(), path, options)
    """
    with matplotlib.rc_context(
        {"svg.hashsalt": options.svg_hashsalt, "svg.fonttype": "path"}
    ):
        figure = chart.draw(table, theme)
        figure.savefig(
            path,
            format=options.image_format,
            dpi=options.png_dpi if options.image_format == "png" else "figure",
            facecolor=theme.surface,
            metadata={"Date": None} if options.image_format == "svg" else None,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Redraw ``experiments-explained/`` from the registry in ``experiments/``.

    Args:
        argv: The command-line arguments, without the program's name.

    Returns:
        The exit status, 0.

    Example:
        >>> main(["--image-format", "png", "--out", "/tmp/chart-previews"])
        0
    """
    parser = argparse.ArgumentParser(
        prog="python -m experiment_charts",
        description="Redraw experiments-explained/ from the registry in experiments/.",
    )
    parser.add_argument("--registry", type=Path, default=Path("experiments"))
    parser.add_argument("--out", type=Path, default=Path("experiments-explained"))
    parser.add_argument("--image-format", choices=("svg", "png"), default="svg")
    arguments = parser.parse_args(argv)
    options = ChartOutput(arguments.registry, arguments.out, arguments.image_format)
    written = write_experiment_charts(options)
    print(f"wrote {len(written)} files under {options.output_dir}")
    return 0
