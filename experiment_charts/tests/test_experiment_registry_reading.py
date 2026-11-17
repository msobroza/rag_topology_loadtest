"""The registry reader, on a tiny registry written into ``tmp_path``."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiment_charts.registry_reading import (
    RunRow,
    number,
    parse_trial_key,
    read_experiment_registry,
)

RESULTS = (
    "run,experiment,users,type,name,requests,failures,requests_per_s,"
    "p50_ms,p95_ms,p99_ms,max_ms\n"
    "tiny-u4,tiny,4,POST,/retrieve,111,3,0.934,3600,4100,4500,5008\n"
    "tiny-u4,tiny,4,STAGE,/retrieve:embed,111,0,0.934,170,320,420,427\n"
)
ENTRIES = """\
common:
  calibration: &calibration {chosen: {threads: 8, batch_size: 2}}
experiments:
  - id: tiny
    calibration: *calibration
    runs: [tiny-u4]
"""


def tiny_registry(root: Path) -> Path:
    sitting = root / "2026-01-01-tiny-sitting"
    sitting.mkdir()
    (sitting / "results.csv").write_text(RESULTS, encoding="utf-8")
    (sitting / "experiments.yaml").write_text(ENTRIES, encoding="utf-8")
    return root


def test_rows_come_back_typed(tmp_path: Path) -> None:
    registry = read_experiment_registry(tiny_registry(tmp_path))
    assert registry.route_row("tiny", 4) == RunRow(
        sitting="2026-01-01-tiny-sitting",
        run="tiny-u4",
        experiment="tiny",
        users=4,
        row_type="POST",
        name="/retrieve",
        requests=111,
        failures=3,
        requests_per_s=0.934,
        p50_ms=3600,
        p95_ms=4100,
    )
    assert [row.name for row in registry.stage_rows("tiny", 4)] == ["/retrieve:embed"]


def test_entries_come_back_with_their_anchors_resolved(tmp_path: Path) -> None:
    registry = read_experiment_registry(tiny_registry(tmp_path))
    chosen = registry.entry_value("tiny", "calibration", "chosen")
    assert chosen == {"threads": 8, "batch_size": 2}
    assert registry.entry("tiny")["runs"] == ["tiny-u4"]


def test_a_missing_row_names_the_experiment_and_the_user_count(tmp_path: Path) -> None:
    registry = read_experiment_registry(tiny_registry(tmp_path))
    with pytest.raises(LookupError, match=r"tiny at 8 users.*user counts \[4\]"):
        registry.route_row("tiny", 8)


def test_a_missing_entry_or_key_says_what_exists(tmp_path: Path) -> None:
    registry = read_experiment_registry(tiny_registry(tmp_path))
    with pytest.raises(LookupError, match="no entry 'other'; the entries are: tiny"):
        registry.entry("other")
    with pytest.raises(LookupError, match=r"no 'trials' in \['chosen'\]"):
        registry.entry_value("tiny", "calibration", "trials")


def test_trial_keys_are_read_into_their_numbers() -> None:
    assert parse_trial_key("batch_20") == {"batch": 20}
    assert parse_trial_key("streams_2_threads_8") == {"streams": 2, "threads": 8}
    with pytest.raises(ValueError, match="'trials_at'"):
        parse_trial_key("trials_at")


def test_a_number_is_a_number_and_nothing_else() -> None:
    assert number(3) == 3.0
    with pytest.raises(TypeError, match="'fast'"):
        number("fast")
