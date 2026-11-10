"""``rag-compare``: locust ``*_stats.csv`` files in, one markdown table per row out.

Example::

    rag-compare results/monolith results/split-all --out comparison.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .request_plans import STAGE_REQUEST_TYPE

# --- rag-compare: locust *_stats.csv files -> one markdown table per row ------


@dataclass(frozen=True)
class RowStats:
    requests: int
    failures: int
    rps: float
    p50: float
    p95: float
    p99: float


def load_stats(prefix: Path) -> dict[str, RowStats]:
    """Read ``<prefix>_stats.csv`` keyed by ``"<Type> <Name>"`` or ``"Aggregated"``."""
    rows: dict[str, RowStats] = {}
    with Path(f"{prefix}_stats.csv").open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            key = (
                "Aggregated"
                if record["Name"] == "Aggregated"
                else f"{record['Type']} {record['Name']}"
            )
            rows[key] = RowStats(
                requests=int(float(record["Request Count"])),
                failures=int(float(record["Failure Count"])),
                rps=float(record["Requests/s"]),
                p50=_number(record["50%"]),
                p95=_number(record["95%"]),
                p99=_number(record["99%"]),
            )
    return rows


def _number(cell: str) -> float:
    return 0.0 if cell.strip() in ("", "N/A") else float(cell)  # N/A: zero-request rows


def render_markdown(
    runs: Mapping[str, Mapping[str, RowStats]],
    targets: Mapping[str, Any] | None = None,
) -> str:
    """One table per row key across all runs; Δp95 is relative to the first run.

    ``targets`` maps a run name to its parsed ``_target.json``, when it has one.

    Example::

        render_markdown({"monolith": load_stats(Path("results/monolith"))})
    """
    labels = list(runs)
    baseline = labels[0]
    keys = sorted({k for stats in runs.values() for k in stats}, key=_row_order)
    lines = ["# Locust run comparison", "", *_intro_lines(labels, targets or {})]
    for key in keys:
        lines += [
            f"### {key}",
            "",
            f"| run | requests | fail % | RPS | p50 | p95 | p99 | Δp95 vs {baseline} |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        base = runs[baseline].get(key)
        for label in labels:
            row = runs[label].get(key)
            delta = _delta(row, base, label == baseline)
            lines.append(f"| {label} | {_cells(row)} | {delta} |")
        lines.append("")
    return "\n".join(lines)


def _intro_lines(labels: Sequence[str], targets: Mapping[str, Any]) -> list[str]:
    """The run list, with the topology and run label each run saw, then a warning
    for any run that saw more than one pair (its numbers mix configurations)."""
    titles, warnings = [], []
    for label in labels:
        seen = targets.get(label, {}).get("seen", [])
        pairs = "; ".join(
            f"topology `{s['topology']}`, run label `{s['run_label']}`" for s in seen
        )
        titles.append(f"{label} ({pairs})" if pairs else label)
        if len(seen) > 1:
            warnings += [
                f"> **Warning:** run `{label}` saw {len(seen)} topology/run label "
                "pairs; its numbers mix configurations.",
                "",
            ]
    intro = f"Runs: {', '.join(titles)}. Δp95 is relative to `{labels[0]}`."
    return [intro, "", *warnings]


def _row_order(key: str) -> tuple[int, str]:
    rank = (
        2
        if key == "Aggregated"
        else 1 if key.startswith(f"{STAGE_REQUEST_TYPE} ") else 0
    )
    return rank, key


def _cells(row: RowStats | None) -> str:
    if row is None:
        return " | ".join(["n/a"] * 6)
    fail = f"{100.0 * row.failures / row.requests:.1f}%" if row.requests else "n/a"
    return (
        f"{row.requests} | {fail} | {row.rps:.1f} | "
        f"{row.p50:.0f} | {row.p95:.0f} | {row.p99:.0f}"
    )


def _delta(row: RowStats | None, base: RowStats | None, is_baseline: bool) -> str:
    if is_baseline:
        return "—"
    if row is None or base is None:
        return "n/a"
    delta = row.p95 - base.p95
    pct = f"{100.0 * delta / base.p95:+.1f}%" if base.p95 else "n/a"
    return f"{delta:+.1f} ms ({pct})"


def compare_main(argv: Sequence[str] | None = None) -> int:
    """``rag-compare results/monolith results/split-all [--out comparison.md]``."""
    parser = argparse.ArgumentParser(
        prog="rag-compare", description="Compare locust runs."
    )
    parser.add_argument("prefixes", nargs="+", type=Path, help="locust --csv prefixes")
    parser.add_argument(
        "--out", type=Path, help="write markdown here (default: stdout)"
    )
    args = parser.parse_args(argv)
    runs = {p.name: load_stats(p) for p in args.prefixes}
    record_files = {p.name: Path(f"{p}_target.json") for p in args.prefixes}
    targets = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in record_files.items()
        if path.exists()
    }
    markdown = render_markdown(runs, targets)
    if args.out is None:
        print(markdown)
    else:
        args.out.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0
