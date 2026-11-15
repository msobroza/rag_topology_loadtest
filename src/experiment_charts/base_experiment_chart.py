"""The port every chart implements: select its numbers, then draw them.

``table`` is pure (the registry in, the numbers out) so the tests check a chart's
figures without drawing it; ``draw`` reads only the table, so the committed CSV
holds exactly what the figure shows.

Example:
    >>> chart = experiment_charts()[0]
    >>> figure = chart.draw(chart.table(registry), light_theme())
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from matplotlib.figure import Figure

from .chart_table import ChartTable
from .chart_theme import ChartTheme
from .registry_reading import ExperimentRegistry


class BaseExperimentChart(ABC):
    """One chart of ``experiments-explained/``: its numbers and its drawing."""

    @property
    @abstractmethod
    def slug(self) -> str:
        """The chart's file name without extension, e.g. ``"08-overload"``."""

    @property
    @abstractmethod
    def title(self) -> str:
        """The chart's title, a plain sentence saying what it plots."""

    @abstractmethod
    def table(self, registry: ExperimentRegistry) -> ChartTable:
        """Select and derive the chart's numbers from the registry.

        Args:
            registry: The experiment registry.

        Returns:
            One row per mark, with the experiment id of each.
        """

    @abstractmethod
    def draw(self, table: ChartTable, theme: ChartTheme) -> Figure:
        """Draw the chart from its table.

        Args:
            table: What ``table`` returned.
            theme: The light or the dark theme.

        Returns:
            The figure, ready to save.
        """
