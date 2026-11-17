"""From responses to the ``## Tokens`` table: one row per run and stage.

The cases of the review of 2026-11-17: 100 ``/query`` answers of 142
completion tokens written in 14.2 s, some after a rewrite of 31 tokens in 2 s,
in a run of 400 s, with or without 300 ``/retrieve`` requests that ran the
rewrite too. One sum over the stages misstated three columns once a ``/query``
ran its rewrite.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from query_processing.loadtest.qp_compare import (
    AGGREGATED_ROW_KEY,
    RowStats,
    render_markdown,
)
from query_processing.loadtest.response_checks import judge_response
from query_processing.loadtest.target_record import (
    StageTokenTotals,
    count_spent_tokens,
    read_target_record,
    write_target_record,
)

REWRITE = {"prompt": 412, "completion": 31}
GENERATE = {"prompt": 1890, "completion": 142}
REWRITE_MS, GENERATE_MS = 2000.0, 14200.0
ANSWERS, RUN_SECONDS = 100, 400.0


def _query_body(rewritten: bool) -> str:
    """A ``/query`` answer as the app writes it; a first message skips rewrite."""
    timings = {"rewrite": REWRITE_MS if rewritten else 0.0, "embed": 30.0}
    timings |= {"generate": GENERATE_MS, "total": GENERATE_MS + 60}
    tokens = {"rewrite": REWRITE} if rewritten else {}
    return json.dumps(
        {"timings_ms": timings, "tokens": tokens | {"generate": GENERATE}}
    )


RETRIEVE_BODY = json.dumps(
    {"timings_ms": {"rewrite": REWRITE_MS}, "tokens": {"rewrite": REWRITE}}
)


def _token_rows(tmp_path: Path, bodies: list[tuple[str, str]]) -> dict[str, list[str]]:
    """The run's ``## Tokens`` rows by stage, its responses judged and recorded."""
    totals: dict[str, StageTokenTotals] = {}
    for endpoint, body in bodies:
        count_spent_tokens(totals, judge_response(endpoint, 200, body).tokens)
    write_target_record(str(tmp_path / "run"), "rag", Counter(), totals)
    record = read_target_record(tmp_path / "run_target.json")
    assert record is not None
    requests = len(bodies)
    # Locust's Aggregated row: every request of the run over its 400 s.
    stats = {AGGREGATED_ROW_KEY: RowStats(requests, 0, requests / RUN_SECONDS, 0, 0, 0)}
    lines = render_markdown({"rag": stats}, {"rag": record}).splitlines()
    table = lines[lines.index("## Tokens") :]
    cells = [_cells(line) for line in table if line.startswith("| rag |")]
    return {stage: rest for _run, stage, *rest in cells}


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip("| ").split(" | ")]


def _queries(rewritten: int) -> list[tuple[str, str]]:
    return [("/query", _query_body(i < rewritten)) for i in range(ANSWERS)]


def test_with_no_history_the_one_row_reads_what_mains_table_read(
    tmp_path: Path,
) -> None:
    rows = _token_rows(tmp_path, _queries(rewritten=0))

    assert rows == {"/query:generate": ["100", "1890.0", "142.0", "10.0", "35.5"]}


def test_a_rewritten_answer_keeps_its_answers_figures(tmp_path: Path) -> None:
    rows = _token_rows(tmp_path, _queries(rewritten=ANSWERS))

    assert rows == {
        "/query:generate": ["100", "1890.0", "142.0", "10.0", "35.5"],
        "/query:rewrite": ["100", "412.0", "31.0", "15.5", "7.8"],
    }


def test_half_the_answers_rewritten_count_half_the_rewrites(tmp_path: Path) -> None:
    rows = _token_rows(tmp_path, _queries(rewritten=ANSWERS // 2))

    assert rows["/query:rewrite"] == ["50", "412.0", "31.0", "15.5", "3.9"]
    assert rows["/query:generate"] == ["100", "1890.0", "142.0", "10.0", "35.5"]


def test_the_rewrites_of_retrieve_count_and_the_rows_add_up_to_what_the_llm_wrote(
    tmp_path: Path,
) -> None:
    retrieves = [("/retrieve", RETRIEVE_BODY)] * 300

    rows = _token_rows(tmp_path, _queries(rewritten=ANSWERS) + retrieves)

    assert rows == {
        "/query:generate": ["100", "1890.0", "142.0", "10.0", "35.5"],
        "/query:rewrite": ["100", "412.0", "31.0", "15.5", "7.8"],
        "/retrieve:rewrite": ["300", "412.0", "31.0", "15.5", "23.2"],
    }
    record = read_target_record(tmp_path / "run_target.json")
    assert record is not None
    assert record.tokens is not None
    # Each row's tokens served is its stage's completion over the run's 400 s.
    written = sum(t.completion for t in record.tokens.values())
    assert written / RUN_SECONDS == pytest.approx(66.5)


def test_a_query_whose_generate_reported_no_counts_adds_to_its_rewrite_alone(
    tmp_path: Path,
) -> None:
    body = json.dumps(
        {
            "timings_ms": {"rewrite": REWRITE_MS, "generate": GENERATE_MS},
            "tokens": {"rewrite": REWRITE, "generate": {"prompt": 1890}},
        }
    )

    rows = _token_rows(tmp_path, [("/query", body)])

    assert list(rows) == ["/query:rewrite"]
    assert rows["/query:rewrite"][:3] == ["1", "412.0", "31.0"]
