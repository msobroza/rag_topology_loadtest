"""``rag-compare``: locust ``*_stats.csv`` files in, one markdown table per row out.

Example:
    $ rag-compare results/monolith results/split-all --out comparison.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .response_checks import STAGE_REQUEST_TYPE
from .target_record import RECORD_SUFFIX, TargetRecord, read_target_record

# --- rag-compare: locust *_stats.csv files -> one markdown table per row ------


@dataclass(frozen=True)
class RowStats:
    """One row of a locust stats CSV: counts, throughput, latency percentiles in ms."""

    requests: int
    failures: int
    rps: float
    p50: float
    p95: float
    p99: float


def load_stats(prefix: Path) -> dict[str, RowStats]:
    """Read ``<prefix>_stats.csv`` keyed by ``"<Type> <Name>"`` or ``"Aggregated"``.

    An empty or ``N/A`` percentile, as locust writes for a zero-request row, reads
    as 0.

    Args:
        prefix: The locust ``--csv`` prefix of the run.

    Returns:
        The stats of every row of the file, by row key.
    """
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
    targets: Mapping[str, TargetRecord] | None = None,
) -> str:
    """Render one table per row key across all runs; Δp95 is relative to the first run.

    Args:
        runs: The stats of each run by row key, keyed by run name; the first run
            is the baseline.
        targets: Maps a run name to its target record, when it has one; ``None``
            when no run has one.

    Returns:
        The markdown: a title, the run list with the topology and run label each
        run saw, then one table per row key, request rows first, STAGE rows next
        and ``Aggregated`` last.

    Example:
        >>> render_markdown({"monolith": load_stats(Path("results/monolith"))})
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


# The run list, with the topology and run label each run saw, then a warning
# for any run that saw more than one pair (its numbers mix configurations).
def _intro_lines(
    labels: Sequence[str], targets: Mapping[str, TargetRecord]
) -> list[str]:
    titles, warnings = [], []
    for label in labels:
        seen = targets[label].seen if label in targets else []
        pairs = "; ".join(
            f"topology `{s.topology}`, run label `{s.run_label}`" for s in seen
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
    """Run ``rag-compare results/monolith results/split-all [--out comparison.md]``.

    Each prefix's ``_target.json`` is read when it exists; runs are named by the
    last component of their prefix.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status, 0, after printing the markdown or writing it to ``--out``.

    Raises:
        ValueError: If a ``_target.json`` is not a target record; the message
            names the file.
    """
    parser = argparse.ArgumentParser(
        prog="rag-compare", description="Compare locust runs."
    )
    parser.add_argument("prefixes", nargs="+", type=Path, help="locust --csv prefixes")
    parser.add_argument(
        "--out", type=Path, help="write markdown here (default: stdout)"
    )
    args = parser.parse_args(argv)
    runs = {p.name: load_stats(p) for p in args.prefixes}
    markdown = render_markdown(runs, _target_records(args.prefixes))
    if args.out is None:
        print(markdown)
    else:
        args.out.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _target_records(prefixes: Sequence[Path]) -> dict[str, TargetRecord]:
    # Each run's target record by run name; a run that left none is skipped.
    records = {
        p.name: read_target_record(Path(f"{p}{RECORD_SUFFIX}")) for p in prefixes
    }
    return {name: record for name, record in records.items() if record is not None}
