"""Client side of the load test: every Locust user decision, and ``rag-compare``.

Values in, decisions out: the run's configuration, what each user sends, what a
response becomes in the statistics (a failure, or STAGE rows), and the target
record. ``locustfile.py`` only does Locust I/O around these, so they are tested
with plain calls and without gevent monkey-patching the interpreter.

Example::

    plan = plan_rag_request("/query", rng)
    verdict = judge_response(plan.path, resp.status_code, resp.text or "")
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .corpus import chunk, synthetic_corpus, synthetic_questions
from .models import STAGES, TOP_K_RERANK, TOP_K_RETRIEVE
from .v3_contract import EMBEDDINGS_PATH, RERANK_PATH, EmbeddingsRequest, RerankRequest

QUESTIONS: list[str] = synthetic_questions(50)

# --- configuration: the environment, read once ---------------------------------

LOAD_TARGETS: tuple[str, ...] = ("rag", "rerank", "embeddings")


@dataclass(frozen=True)
class LoadTestConfig:
    """Everything a run reads from its environment, validated up front."""

    target: str
    query_weight: int
    retrieve_weight: int
    step_shape: bool
    auth_headers: Mapping[str, str]


def read_loadtest_config(env: Mapping[str, str]) -> LoadTestConfig:
    """Read and validate the ``RAG_LOADTEST_*`` knobs (and Domino auth) once.

    Example::

        read_loadtest_config(os.environ).target  # "rag" unless overridden
    """
    target = env.get("RAG_LOADTEST_TARGET", "rag")
    if target not in LOAD_TARGETS:
        allowed = ", ".join(LOAD_TARGETS)
        raise ValueError(f"RAG_LOADTEST_TARGET={target!r}; expected one of {allowed}")
    return LoadTestConfig(
        target=target,
        query_weight=_task_weight(env, "RAG_LOADTEST_QUERY_WEIGHT", 1),
        retrieve_weight=_task_weight(env, "RAG_LOADTEST_RETRIEVE_WEIGHT", 3),
        step_shape=env.get("RAG_LOADTEST_SHAPE") == "step",
        auth_headers=_auth_headers(env),
    )


def _task_weight(env: Mapping[str, str], variable: str, default: int) -> int:
    raw = env.get(variable)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{variable}={raw!r}; expected an integer") from None


def _auth_headers(env: Mapping[str, str]) -> dict[str, str]:
    """Domino auth for callers outside a run: a bearer token wins over an API key."""
    if env.get("RAG_LOADTEST_BEARER_TOKEN"):
        return {"Authorization": f"Bearer {env['RAG_LOADTEST_BEARER_TOKEN']}"}
    if env.get("DOMINO_API_KEY"):
        return {"X-Domino-Api-Key": env["DOMINO_API_KEY"]}
    return {}


# --- what to send ---------------------------------------------------------------

# The same candidate count the workflow sends to its reranker.
_RERANK_DOCUMENTS: list[str] = [
    p.text for d in synthetic_corpus(TOP_K_RETRIEVE) for p in chunk(d)
][:TOP_K_RETRIEVE]


@dataclass(frozen=True)
class RequestPlan:
    """One request to send: where, with which JSON body, under which stats name."""

    path: str
    payload: dict[str, object]
    stats_name: str


def plan_rag_request(endpoint: str, rng: random.Random) -> RequestPlan:
    """A workflow-app request: one random question to ``/query`` or ``/retrieve``.

    Example::

        plan_rag_request("/query", random.Random(7)).payload  # {"question": "..."}
    """
    return RequestPlan(endpoint, {"question": rng.choice(QUESTIONS)}, endpoint)


def plan_rerank_request(alias: str, rng: random.Random) -> RequestPlan:
    """A model-server rerank request shaped like the workflow's rerank stage.

    Example::

        plan_rerank_request("bge-reranker-base", random.Random(7)).path  # /v3/rerank
    """
    request = RerankRequest(
        model=alias,
        query=rng.choice(QUESTIONS),
        documents=list(_RERANK_DOCUMENTS),
        top_n=TOP_K_RERANK,
    )
    return RequestPlan(RERANK_PATH, request.to_wire(), RERANK_PATH)


def plan_embeddings_request(alias: str, rng: random.Random) -> RequestPlan:
    """A model-server embeddings request shaped like the workflow's embed stage.

    Example::

        plan_embeddings_request("bge-small-en-v1.5", random.Random(7)).path
    """
    request = EmbeddingsRequest(model=alias, input=[rng.choice(QUESTIONS)])
    return RequestPlan(EMBEDDINGS_PATH, request.to_wire(), EMBEDDINGS_PATH)


# --- what to record -------------------------------------------------------------

# The request type of STAGE rows, used both when firing them and by rag-compare.
STAGE_REQUEST_TYPE = "STAGE"


@dataclass(frozen=True)
class ResponseVerdict:
    """What one response becomes in the statistics: a failure, or STAGE rows
    (``(name, ms)`` pairs) to fire."""

    failure: str | None
    stage_rows: tuple[tuple[str, float], ...] = ()


def judge_response(endpoint: str, status: int, body_text: str) -> ResponseVerdict:
    """Decide how a response is recorded; only a 200 with a JSON object succeeds.

    Example::

        judge_response("/query", 200, '{"timings_ms": {"embed": 4}}').stage_rows
        # (("/query:embed", 4.0),)
    """
    if status != 200:
        return ResponseVerdict(failure=f"HTTP {status}: {body_text[:200]}")
    try:
        body = json.loads(body_text)
    except ValueError:
        body = None
    if not isinstance(body, dict):
        message = f"HTTP 200 but the body is not a JSON object: {body_text[:200]}"
        return ResponseVerdict(failure=message)
    return ResponseVerdict(failure=None, stage_rows=_stage_rows(endpoint, body))


def _stage_rows(
    endpoint: str, body: Mapping[str, Any]
) -> tuple[tuple[str, float], ...]:
    """One row per stage timed above zero, in ``STAGES`` order (``total`` excluded);
    none when ``timings_ms`` is missing or malformed."""
    try:
        timings_ms = {str(k): float(v) for k, v in body["timings_ms"].items()}
    except (AttributeError, KeyError, TypeError, ValueError):
        return ()
    return tuple(
        (f"{endpoint}:{s}", timings_ms[s]) for s in STAGES if timings_ms.get(s, 0) > 0
    )


# --- the target record: what the target said about itself during a run -------


def target_pair(
    status: int, headers: Mapping[str, str] | None
) -> tuple[str, str] | None:
    """``(topology, run label)`` a response reported; ``None`` when none came back.

    Example::

        target_pair(200, {"X-RAG-Topology": "monolith"})  # ("monolith", "")
    """
    if status == 0:  # Locust's status for a connection error: no response at all
        return None
    headers = headers or {}
    return headers.get("X-RAG-Topology", ""), headers.get("X-RAG-Run-Label", "")


def write_target_record(
    csv_prefix: str | None, target: str, counts: Mapping[tuple[str, str], int]
) -> None:
    """Write ``<csv_prefix>_target.json``, most responses first; nothing without
    ``--csv``. Model servers send no topology headers, so their ``seen`` is empty.

    Example::

        write_target_record("results/monolith", "rag", Counter({("monolith", "m"): 3}))
    """
    if not csv_prefix:
        return
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    seen = [
        {"topology": topology, "run_label": label, "responses": n}
        for (topology, label), n in ordered
        if target == "rag"
    ]
    document = {"target": target, "seen": seen}
    Path(f"{csv_prefix}_target.json").write_text(json.dumps(document, indent=2))


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
