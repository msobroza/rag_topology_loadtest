"""Client side of the load test: Locust-free helpers and the ``rag-compare`` command.

Everything ``locustfile.py`` needs that does not touch locust lives here so it
can be unit-tested without gevent monkey-patching the interpreter.

Example::

    for name, ms in stage_events("/query", parse_timings(body)):
        events.request.fire(request_type="STAGE", name=name, response_time=ms, ...)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .corpus import synthetic_questions
from .models import STAGES

QUESTIONS: list[str] = synthetic_questions(50)


def auth_headers(env: Mapping[str, str]) -> dict[str, str]:
    """Domino auth for callers outside a run: a bearer token wins over an API key."""
    if env.get("RAG_LOADTEST_BEARER_TOKEN"):
        return {"Authorization": f"Bearer {env['RAG_LOADTEST_BEARER_TOKEN']}"}
    if env.get("DOMINO_API_KEY"):
        return {"X-Domino-Api-Key": env["DOMINO_API_KEY"]}
    return {}


def task_weights(env: Mapping[str, str]) -> tuple[int, int]:
    """``(query, retrieve)`` task weights from the environment; defaults ``(1, 3)``."""
    return (
        int(env.get("RAG_LOADTEST_QUERY_WEIGHT", 1)),
        int(env.get("RAG_LOADTEST_RETRIEVE_WEIGHT", 3)),
    )


def parse_timings(body: Mapping[str, Any]) -> dict[str, float]:
    """``body["timings_ms"]`` as floats, or ``{}`` when missing or malformed."""
    try:
        return {str(k): float(v) for k, v in body.get("timings_ms", {}).items()}
    except (AttributeError, TypeError, ValueError):
        return {}


def stage_events(
    endpoint: str, timings_ms: Mapping[str, float]
) -> list[tuple[str, float]]:
    """Extra locust events, one per stage with a non-zero time (``total`` excluded).

    Example::

        stage_events("/query", {"embed": 4.0, "generate": 0.0, "total": 9.0})
        # [("/query:embed", 4.0)]
    """
    return [
        (f"{endpoint}:{s}", timings_ms[s]) for s in STAGES if timings_ms.get(s, 0) > 0
    ]


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


def render_markdown(runs: Mapping[str, Mapping[str, RowStats]]) -> str:
    """One table per row key across all runs; Δp95 is relative to the first run."""
    labels = list(runs)
    baseline = labels[0]
    keys = sorted({k for stats in runs.values() for k in stats}, key=_row_order)
    intro = f"Runs: {', '.join(labels)}. Δp95 is relative to `{baseline}`."
    lines = ["# Locust run comparison", "", intro, ""]
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


def _row_order(key: str) -> tuple[int, str]:
    rank = 2 if key == "Aggregated" else 1 if key.startswith("STAGE ") else 0
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
    markdown = render_markdown({p.name: load_stats(p) for p in args.prefixes})
    if args.out is None:
        print(markdown)
    else:
        args.out.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0
