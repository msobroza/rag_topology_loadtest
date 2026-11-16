"""Tests for ``rag-compare`` (``rag_load_test.loadtest.rag_compare``)."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from rag_load_test.loadtest.rag_compare import (
    RowStats,
    compare_main,
    load_stats,
    render_markdown,
)
from rag_load_test.loadtest.target_record import (
    TargetRecord,
    TokenTotals,
    read_target_record,
    write_target_record,
)

STATS_HEADER = [
    "Type",
    "Name",
    "Request Count",
    "Failure Count",
    "Median Response Time",
    "Average Response Time",
    "Min Response Time",
    "Max Response Time",
    "Average Content Size",
    "Requests/s",
    "Failures/s",
    "50%",
    "66%",
    "75%",
    "80%",
    "90%",
    "95%",
    "98%",
    "99%",
    "99.9%",
    "99.99%",
    "100%",
]

# --- rag-compare ---

_ROW_KEYS = ("Type", "Name", "Request Count", "Failure Count", "Requests/s")
MONOLITH = (
    ("POST", "/query", 100, 2, 4.5, 20, 30, 40),
    ("STAGE", "/query:embed", 98, 0, 4.4, 3, 5, 6),
    ("", "Aggregated", 198, 2, 8.9, 10, 30, 40),
)
SPLIT = (
    ("POST", "/query", 100, 0, 5.0, 25, 45, 60),
    ("", "Aggregated", 100, 0, 5.0, 25, 45, 60),
)


def _stats_csv(prefix: Path, *rows: tuple[Any, ...]) -> Path:
    """Write ``<prefix>_stats.csv``; a row is (Type, Name, requests, failures, rps, p50, p95, p99)."""
    with Path(f"{prefix}_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATS_HEADER)
        writer.writeheader()
        for row in rows:
            keys = (*_ROW_KEYS, "50%", "95%", "99%")
            cells = dict(zip(keys, map(str, row), strict=True))
            writer.writerow({**dict.fromkeys(STATS_HEADER, "0"), **cells})
    return prefix


def test_load_stats_keys_rows_and_zeroes_na_percentiles(tmp_path: Path) -> None:
    idle = ("POST", "/retrieve", 0, 0, 0.0, "N/A", "N/A", "N/A")
    stats = load_stats(_stats_csv(tmp_path / "monolith", *MONOLITH, idle))
    expected_keys = {
        "POST /query",
        "STAGE /query:embed",
        "POST /retrieve",
        "Aggregated",
    }
    assert set(stats) == expected_keys
    assert stats["POST /query"] == RowStats(100, 2, 4.5, 20.0, 30.0, 40.0)
    assert stats["Aggregated"].requests == 198
    assert stats["POST /retrieve"] == RowStats(0, 0, 0.0, 0.0, 0.0, 0.0)


def test_render_markdown_orders_tables_and_computes_delta_p95(tmp_path: Path) -> None:
    runs = {
        "monolith": load_stats(_stats_csv(tmp_path / "monolith", *MONOLITH)),
        "split-all": load_stats(_stats_csv(tmp_path / "split-all", *SPLIT)),
    }
    markdown = render_markdown(runs)
    headings = [line for line in markdown.splitlines() if line.startswith("### ")]
    assert headings == ["### POST /query", "### STAGE /query:embed", "### Aggregated"]
    assert "Δp95 is relative to `monolith`" in markdown
    assert "| monolith | 100 | 2.0% | 4.5 | 20 | 30 | 40 | — |" in markdown
    split_row = "| split-all | 100 | 0.0% | 5.0 | 25 | 45 | 60 | +15.0 ms (+50.0%) |"
    assert split_row in markdown
    assert "| split-all | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in markdown


def test_compare_main_writes_out_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monolith = _stats_csv(tmp_path / "monolith", *MONOLITH)
    split = _stats_csv(tmp_path / "split-all", *SPLIT)
    out = tmp_path / "comparison.md"
    assert compare_main([str(monolith), str(split), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"wrote {out}" in captured.err
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Locust run comparison")
    assert "Runs: monolith, split-all." in text
    assert "+15.0 ms (+50.0%)" in text


def _target_json(prefix: Path, target: str, *seen: tuple[str, str, int]) -> None:
    """Write ``<prefix>_target.json``; a sighting is (topology, run label, responses)."""
    write_target_record(str(prefix), target, Counter({(t, r): n for t, r, n in seen}))


def test_compare_names_topology_and_run_label_from_target_records(
    tmp_path: Path,
) -> None:
    monolith = _stats_csv(tmp_path / "monolith", *MONOLITH)
    split = _stats_csv(tmp_path / "split-all", *SPLIT)
    _target_json(monolith, "rag", ("monolith", "monolith-cpu", 198))
    out = tmp_path / "comparison.md"
    assert compare_main([str(monolith), str(split), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    intro = (
        "Runs: monolith (topology `monolith`, run label `monolith-cpu`), split-all. "
        "Δp95 is relative to `monolith`."
    )
    assert intro in text.splitlines()
    assert "Warning" not in text


def test_compare_warns_when_a_run_saw_more_than_one_pair(tmp_path: Path) -> None:
    split = _stats_csv(tmp_path / "split-all", *SPLIT)
    _target_json(split, "rag", ("split-all", "a", 60), ("split-reranker", "b", 40))
    markdown = render_markdown(
        {"split-all": load_stats(split)},
        {"split-all": read_target_record(Path(f"{split}_target.json"))},
    )
    intro = (
        "Runs: split-all (topology `split-all`, run label `a`; "
        "topology `split-reranker`, run label `b`). Δp95 is relative to `split-all`."
    )
    lines = markdown.splitlines()
    assert intro in lines
    warning = (
        "> **Warning:** run `split-all` saw 2 topology/run label pairs; "
        "its numbers mix configurations."
    )
    assert lines[lines.index(intro) + 2] == warning


def test_compare_lists_a_model_server_run_by_name_alone(tmp_path: Path) -> None:
    ovms = _stats_csv(tmp_path / "ovms-rerank", *SPLIT)
    _target_json(ovms, "rerank")
    out = tmp_path / "comparison.md"
    assert compare_main([str(ovms), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Runs: ovms-rerank. Δp95 is relative to `ovms-rerank`." in text.splitlines()


def test_compare_refuses_a_malformed_target_record_naming_its_file(
    tmp_path: Path,
) -> None:
    monolith = _stats_csv(tmp_path / "monolith", *MONOLITH)
    record = Path(f"{monolith}_target.json")
    record.write_text('{"target": "rag", "seen": "everything"}', encoding="utf-8")

    with pytest.raises(ValueError, match="expected a target record") as caught:
        compare_main([str(monolith)])

    assert str(caught.value).startswith(f"{record}: seen: ")


# --- tokens ---

OVMS_QUERY = (
    ("POST", "/query", 22, 0, 0.124, 7400, 9000, 9500),
    ("", "Aggregated", 22, 0, 0.124, 7400, 9000, 9500),
)
OVMS_TOKENS = TokenTotals(
    answers=22, prompt=17446, completion=2516, generate_ms=164428.0
)


def _token_runs(tmp_path: Path) -> tuple[dict[str, Any], dict[str, TargetRecord]]:
    """Two runs: ``ovms`` with a token total, ``split-all`` with a record but none."""
    ovms = _stats_csv(tmp_path / "ovms", *OVMS_QUERY)
    split = _stats_csv(tmp_path / "split-all", *SPLIT)
    write_target_record(str(ovms), "rag", Counter(), OVMS_TOKENS)
    write_target_record(str(split), "rag", Counter(), None)
    runs = {"ovms": load_stats(ovms), "split-all": load_stats(split)}
    records = {
        name: read_target_record(Path(f"{prefix}_target.json"))
        for name, prefix in (("ovms", ovms), ("split-all", split))
    }
    return runs, {name: r for name, r in records.items() if r is not None}


def test_a_run_with_tokens_has_a_row_in_the_tokens_table(tmp_path: Path) -> None:
    runs, records = _token_runs(tmp_path)

    lines = render_markdown(runs, records).splitlines()

    start = lines.index("## Tokens")
    table = lines[start:]
    assert table[2] == (
        "| run | answers | prompt tokens per answer | completion tokens per answer "
        "| tokens/s as a caller sees them | tokens/s served |"
    )
    assert "| ovms | 22 | 793.0 | 114.4 | 15.3 | 14.2 |" in table
    assert not any(line.startswith("| split-all |") for line in table)


def test_no_run_with_tokens_means_no_tokens_section(tmp_path: Path) -> None:
    split = _stats_csv(tmp_path / "split-all", *SPLIT)
    write_target_record(str(split), "rag", Counter(), None)
    record = read_target_record(Path(f"{split}_target.json"))
    assert record is not None

    markdown = render_markdown({"split-all": load_stats(split)}, {"split-all": record})

    assert "## Tokens" not in markdown
