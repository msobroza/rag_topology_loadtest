"""What the catalog hands a chart: experiment ids, each with its label.

Example:
    >>> RunPair("8 vCPUs", "v8-e2e-fastapi-fp32", "v8-e2e-ovms-fp32-2streams")
"""

from __future__ import annotations

from dataclasses import dataclass

from .chart_table import Cell
from .registry_reading import ExperimentRegistry


@dataclass(frozen=True)
class LabelledRun:
    """One experiment and the words a chart shows for it."""

    label: str
    experiment: str


@dataclass(frozen=True)
class RunPair:
    """Two experiments compared side by side under one label.

    For a comparison of the model servers ``first`` is the FastAPI model server's
    experiment and ``second`` OVMS's.
    """

    label: str
    first: str
    second: str


SERVERS = ("FastAPI", "OVMS")  # the order of RunPair's first and second


def server_pair_rows(
    registry: ExperimentRegistry, pairs: tuple[RunPair, ...], users: int
) -> list[tuple[Cell, ...]]:
    """Each pair's two route rows: group, server, experiment, users, requests_per_s.

    Args:
        registry: The experiment registry.
        pairs: The model servers' experiments, FastAPI first, under a group label.
        users: The runs' user count.

    Returns:
        Two rows per pair, FastAPI's first.

    Example:
        >>> server_pair_rows(registry, (RunPair("8 vCPUs", fastapi, ovms),), 4)[0][1]
        'FastAPI'
    """
    return [
        (
            pair.label,
            server,
            experiment,
            users,
            registry.route_row(experiment, users).requests_per_s,
        )
        for pair in pairs
        for server, experiment in zip(SERVERS, (pair.first, pair.second), strict=True)
    ]


SERVER_PAIR_COLUMNS = ("group", "server", "experiment", "users", "requests_per_s")
