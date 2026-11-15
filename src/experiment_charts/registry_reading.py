"""Read the experiment registry: every sitting's ``results.csv`` and YAML entries.

The registry's conventions are in ``experiments/README.md``: a sitting is a folder,
``results.csv`` holds one row per row of each run's Locust statistics, and each
``experiments*.yaml`` lists its experiments under ``experiments:``.

Example:
    >>> registry = read_experiment_registry(Path("experiments"))
    >>> registry.route_row("v8-alone-fastapi-fp32", 4).requests_per_s
    1.004
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunRow:
    """One row of a sitting's ``results.csv``, times in milliseconds."""

    sitting: str  # the sitting's folder name
    run: str
    experiment: str
    users: int
    row_type: str  # "POST" (the route), "STAGE", or "" (Locust's Aggregated row)
    name: str
    requests: int
    failures: int
    requests_per_s: float
    p50_ms: int
    p95_ms: int


@dataclass(frozen=True)
class ExperimentRegistry:
    """Every run row and every experiment entry of the registry."""

    rows: tuple[RunRow, ...]
    entries: Mapping[str, Mapping[str, object]]  # experiment id -> its YAML entry

    def route_row(self, experiment: str, users: int) -> RunRow:
        """The ``POST`` row of ``experiment``'s run at ``users`` users.

        Args:
            experiment: The experiment id, e.g. ``"v8-alone-fastapi-fp32"``.
            users: The run's user count.

        Returns:
            The route's row of that run.

        Raises:
            LookupError: The experiment has no run at that user count.

        Example:
            >>> registry.route_row("v8-alone-fastapi-fp32", 4).requests_per_s
            1.004
        """
        rows = self._rows_of(experiment, users, "POST")
        if len(rows) != 1:
            raise LookupError(self._missing(experiment, users, "one POST row", rows))
        return rows[0]

    def stage_rows(self, experiment: str, users: int) -> tuple[RunRow, ...]:
        """The ``STAGE`` rows of ``experiment``'s run at ``users`` users.

        Args:
            experiment: The experiment id, e.g. ``"v8-e2e-fastapi-fp32"``.
            users: The run's user count.

        Returns:
            The stage rows, in the file's order.

        Raises:
            LookupError: The run has no stage row.

        Example:
            >>> [row.name for row in registry.stage_rows("v8-e2e-fastapi-fp32", 4)]
            ['/retrieve:embed', '/retrieve:rerank', '/retrieve:retrieve']
        """
        rows = self._rows_of(experiment, users, "STAGE")
        if not rows:
            raise LookupError(self._missing(experiment, users, "STAGE rows", rows))
        return rows

    def entry(self, experiment: str) -> Mapping[str, object]:
        """The YAML entry of ``experiment``, anchors resolved.

        Args:
            experiment: The experiment id, e.g. ``"v8-alone-fastapi-fp32"``.

        Returns:
            The entry, as parsed.

        Raises:
            LookupError: No YAML file of the registry has that id.

        Example:
            >>> registry.entry("v8-alone-fastapi-fp32")["load"]["users"]
            [1, 4, 8, 16, 32]
        """
        if experiment not in self.entries:
            known = ", ".join(sorted(self.entries))
            raise LookupError(f"no entry {experiment!r}; the entries are: {known}")
        return self.entries[experiment]

    def entry_value(self, experiment: str, *keys: str) -> object:
        """The value under ``keys`` in ``experiment``'s entry.

        Args:
            experiment: The experiment id, e.g. ``"v8-alone-fastapi-fp32"``.
            *keys: The keys from the entry down, e.g. ``"calibration", "chosen"``.

        Returns:
            The value found there.

        Raises:
            LookupError: A key is missing on the way.

        Example:
            >>> registry.entry_value("v8-alone-fastapi-fp32", "calibration", "chosen")
            {'threads': 8, 'batch_size': 2, 'seconds_per_request': 1.24}
        """
        value: object = self.entry(experiment)
        for key in keys:
            if not isinstance(value, Mapping) or key not in value:
                path = ".".join(keys)
                found = sorted(value) if isinstance(value, Mapping) else repr(value)
                raise LookupError(f"{experiment} has no {path}: no {key!r} in {found}")
            value = value[key]
        return value

    def _rows_of(
        self, experiment: str, users: int, row_type: str
    ) -> tuple[RunRow, ...]:
        return tuple(
            row
            for row in self.rows
            if (row.experiment, row.users, row.row_type)
            == (experiment, users, row_type)
        )

    def _missing(
        self, experiment: str, users: int, wanted: str, found: tuple[RunRow, ...]
    ) -> str:
        counts = sorted(
            {row.users for row in self.rows if row.experiment == experiment}
        )
        return (
            f"{experiment} at {users} users: expected {wanted}, found {len(found)}; "
            f"its runs are at user counts {counts}"
        )


def read_experiment_registry(registry_dir: Path) -> ExperimentRegistry:
    """Read every sitting of the registry at ``registry_dir``.

    Args:
        registry_dir: The registry, ``experiments/`` of the repository.

    Returns:
        Every ``*/results.csv`` row and every ``*/experiments*.yaml`` entry.

    Example:
        >>> len(read_experiment_registry(Path("experiments")).entries)
        70
    """
    rows = [
        row
        for path in sorted(registry_dir.glob("*/results.csv"))
        for row in _read_results(path)
    ]
    entries: dict[str, Mapping[str, object]] = {}
    for path in sorted(registry_dir.glob("*/experiments*.yaml")):
        entries |= _read_entries(path)
    return ExperimentRegistry(tuple(rows), entries)


def parse_trial_key(key: str) -> dict[str, int]:
    """Read the numbers out of a trial's key, ``batch_2`` or ``streams_1_threads_8``.

    Args:
        key: The key, ``<name>_<n>`` pairs joined by underscores.

    Returns:
        Each name with its number.

    Raises:
        ValueError: The key is not made of ``<name>_<n>`` pairs.

    Example:
        >>> parse_trial_key("streams_2_threads_8")
        {'streams': 2, 'threads': 8}
    """
    if not re.fullmatch(r"[a-z]+_\d+(_[a-z]+_\d+)*", key):
        raise ValueError(
            f"trial key {key!r} is not like 'threads_8' or 'streams_1_threads_8'"
        )
    return {name: int(number) for name, number in re.findall(r"([a-z]+)_(\d+)", key)}


def number(value: object) -> float:
    """``value`` as a float, if the YAML holds a number there.

    Args:
        value: A value read from an entry.

    Returns:
        The number.

    Raises:
        TypeError: The value is not a number.

    Example:
        >>> number(1.24)
        1.24
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected a number in the registry, found {value!r}")
    return float(value)


def _read_results(path: Path) -> list[RunRow]:
    with path.open(encoding="utf-8", newline="") as results:
        return [_run_row(path.parent.name, line) for line in csv.DictReader(results)]


def _run_row(sitting: str, line: dict[str, str]) -> RunRow:
    return RunRow(
        sitting=sitting,
        run=line["run"],
        experiment=line["experiment"],
        users=int(line["users"]),
        row_type=line["type"],
        name=line["name"],
        requests=int(line["requests"]),
        failures=int(line["failures"]),
        requests_per_s=float(line["requests_per_s"]),
        p50_ms=int(line["p50_ms"]),
        p95_ms=int(line["p95_ms"]),
    )


def _read_entries(path: Path) -> dict[str, Mapping[str, object]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    listed = document.get("experiments") if isinstance(document, dict) else None
    if not isinstance(listed, list):
        raise ValueError(f"{path.name} has no list under 'experiments:'")
    return {str(entry["id"]): entry for entry in listed}
