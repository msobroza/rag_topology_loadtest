"""The marks' colour helpers: each name a table gives gets the theme's colour."""

from __future__ import annotations

from experiment_charts.chart_marks import generator_server_colours
from experiment_charts.chart_theme import dark_theme, light_theme


def test_the_generator_servers_keep_ovms_orange_and_differ() -> None:
    for theme in (light_theme(), dark_theme()):
        colours = generator_server_colours(theme)
        assert list(colours) == ["OVMS", "PyTorch", "vLLM"]
        assert colours["OVMS"] == theme.ovms
        assert len(set(colours.values())) == 3
