"""The target record: the pair a response reports, and the record written and read."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from query_processing.loadtest.response_checks import AnswerTokens
from query_processing.loadtest.target_record import (
    TargetPair,
    TargetRecord,
    TargetSighting,
    TokenTotals,
    read_target_record,
    target_pair,
    write_target_record,
)

# --- target record ---

TOPOLOGY, RUN_LABEL = "X-RAG-Topology", "X-RAG-Run-Label"


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({TOPOLOGY: "monolith", RUN_LABEL: "mono-a"}, ("monolith", "mono-a")),
        ({TOPOLOGY: "monolith"}, ("monolith", "")),
        # A workflow app given no topology alias reports none: None, not a name.
        ({RUN_LABEL: "mono-a"}, (None, "mono-a")),
        # Neither header: not the workflow app answering (a proxy's error page, a
        # model server, a connection error), so there is nothing to count.
        ({"Content-Type": "text/html"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_target_pair(
    headers: dict[str, str] | None, expected: TargetPair | None
) -> None:
    assert target_pair(headers) == expected


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_write_target_record_puts_most_responses_first(tmp_path: Path) -> None:
    counts = Counter(
        {("monolith", "b"): 5, ("split-all", "x"): 9, ("monolith", "a"): 5}
    )
    write_target_record(str(tmp_path / "monolith"), "rag", counts)
    assert _read_json(tmp_path / "monolith_target.json") == {
        "target": "rag",
        "seen": [
            {"topology": "split-all", "run_label": "x", "responses": 9},
            {"topology": "monolith", "run_label": "a", "responses": 5},
            {"topology": "monolith", "run_label": "b", "responses": 5},
        ],
    }


def test_a_target_that_reported_no_topology_is_recorded_with_none(
    tmp_path: Path,
) -> None:
    # A local run (no alias) beside a deployed one, the tie ordered no alias first.
    counts = Counter({("monolith", "m"): 4, (None, "embedder=local"): 4})
    write_target_record(str(tmp_path / "mixed"), "rag", counts)

    path = tmp_path / "mixed_target.json"
    assert _read_json(path)["seen"] == [
        {"topology": None, "run_label": "embedder=local", "responses": 4},
        {"topology": "monolith", "run_label": "m", "responses": 4},
    ]
    record = read_target_record(path)
    assert record is not None
    assert [s.topology for s in record.seen] == [None, "monolith"]


@pytest.mark.parametrize("target", ["rerank", "embeddings"])
def test_write_target_record_model_server_sees_nothing(
    tmp_path: Path, target: str
) -> None:
    write_target_record(str(tmp_path / "ovms"), target, Counter({("", ""): 40}))
    assert _read_json(tmp_path / "ovms_target.json") == {"target": target, "seen": []}


@pytest.mark.parametrize("csv_prefix", [None, ""])
def test_write_target_record_without_csv_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, csv_prefix: str | None
) -> None:
    monkeypatch.chdir(tmp_path)
    write_target_record(csv_prefix, "rag", Counter({("m", "m"): 1}))
    assert list(tmp_path.iterdir()) == []


def test_a_written_record_reads_back_as_the_typed_class(tmp_path: Path) -> None:
    counts = Counter({("split-all", "x"): 9, ("monolith", "a"): 5})
    write_target_record(str(tmp_path / "run"), "rag", counts)

    assert read_target_record(tmp_path / "run_target.json") == TargetRecord(
        target="rag",
        seen=[
            TargetSighting(topology="split-all", run_label="x", responses=9),
            TargetSighting(topology="monolith", run_label="a", responses=5),
        ],
    )


def test_a_record_from_an_earlier_run_still_reads(tmp_path: Path) -> None:
    path = tmp_path / "old_target.json"
    path.write_text(
        '{\n  "target": "rag",\n  "seen": [\n    {\n      "topology": "monolith",\n'
        '      "run_label": "m",\n      "responses": 3\n    }\n  ]\n}'
    )

    record = read_target_record(path)

    assert record is not None
    assert [(s.topology, s.run_label, s.responses) for s in record.seen] == [
        ("monolith", "m", 3)
    ]


def test_a_record_without_tokens_still_reads(tmp_path: Path) -> None:
    path = tmp_path / "old_target.json"
    path.write_text('{"target": "rag", "seen": []}')

    record = read_target_record(path)

    assert record is not None
    assert record.tokens is None


def test_totals_add_one_answer() -> None:
    total = TokenTotals(answers=1, prompt=829, completion=128, llm_ms=7474.0)
    assert total.plus(AnswerTokens(922, 70, 5000.0)) == TokenTotals(
        answers=2, prompt=1751, completion=198, llm_ms=12474.0
    )


@pytest.mark.parametrize(
    "tokens", [None, TokenTotals(answers=0, prompt=0, completion=0, llm_ms=0.0)]
)
def test_a_run_without_tokens_writes_no_tokens_key(
    tmp_path: Path, tokens: TokenTotals | None
) -> None:
    write_target_record(str(tmp_path / "run"), "rag", Counter(), tokens)
    assert "tokens" not in _read_json(tmp_path / "run_target.json")


def test_a_run_with_tokens_writes_and_reads_them_back(tmp_path: Path) -> None:
    tokens = TokenTotals(answers=2, prompt=1751, completion=198, llm_ms=12474.0)
    write_target_record(str(tmp_path / "run"), "rag", Counter(), tokens)

    assert _read_json(tmp_path / "run_target.json")["tokens"] == {
        "answers": 2,
        "prompt": 1751,
        "completion": 198,
        "llm_ms": 12474.0,
    }
    record = read_target_record(tmp_path / "run_target.json")
    assert record is not None
    assert record.tokens == tokens


def test_tokens_leave_a_target_without_topology_recorded_with_none(
    tmp_path: Path,
) -> None:
    # Leaving out a None total must not leave out a None topology alias.
    tokens = TokenTotals(answers=1, prompt=829, completion=128, llm_ms=7474.0)
    counts = Counter({(None, "embedder=local"): 1})
    write_target_record(str(tmp_path / "run"), "rag", counts, tokens)

    assert _read_json(tmp_path / "run_target.json")["seen"] == [
        {"topology": None, "run_label": "embedder=local", "responses": 1}
    ]


def test_a_negative_token_total_is_refused_naming_the_field(tmp_path: Path) -> None:
    path = tmp_path / "bad_target.json"
    path.write_text(
        '{"target": "rag", "seen": [], "tokens": '
        '{"answers": 1, "prompt": -1, "completion": 1, "llm_ms": 1.0}}'
    )

    with pytest.raises(ValueError, match="expected a target record") as caught:
        read_target_record(path)

    assert str(caught.value).startswith(f"{path}: tokens.prompt: ")


@pytest.mark.parametrize("llm_time_key", ["llm_ms", "generate_ms"])
def test_the_llm_time_reads_under_its_name_and_under_mains(
    tmp_path: Path, llm_time_key: str
) -> None:
    # main's load test wrote the LLM's time as generate_ms (pull request 27).
    path = tmp_path / "run_target.json"
    path.write_text(
        '{"target": "rag", "seen": [], "tokens": {"answers": 1, "prompt": 829, '
        f'"completion": 128, "{llm_time_key}": 7474.0}}}}'
    )

    record = read_target_record(path)

    assert record is not None
    assert record.tokens == TokenTotals(
        answers=1, prompt=829, completion=128, llm_ms=7474.0
    )


def test_another_unknown_key_of_the_tokens_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad_target.json"
    path.write_text(
        '{"target": "rag", "seen": [], "tokens": {"answers": 1, "prompt": 1, '
        '"completion": 1, "llm_ms": 1.0, "rerank_ms": 1.0}}'
    )

    with pytest.raises(ValueError, match="expected a target record") as caught:
        read_target_record(path)

    assert str(caught.value).startswith(f"{path}: tokens.rerank_ms: ")


def test_a_missing_record_reads_as_none(tmp_path: Path) -> None:
    assert read_target_record(tmp_path / "absent_target.json") is None


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("not json", "the document"),
        ('{"target": "rag"}', "seen"),
        ('{"target": "ovms", "seen": []}', "target"),
        (
            '{"target": "rag", "seen": [{"topology": "m", "run_label": "m"}]}',
            "seen.0.responses",
        ),
        ('{"target": "rag", "seen": [], "extra": 1}', "extra"),
    ],
    ids=["not-json", "no-seen", "unknown-target", "no-responses", "extra-field"],
)
def test_a_malformed_record_names_the_file_and_the_field(
    tmp_path: Path, text: str, field: str
) -> None:
    path = tmp_path / "bad_target.json"
    path.write_text(text)

    with pytest.raises(ValueError, match="expected a target record") as caught:
        read_target_record(path)

    assert str(caught.value).startswith(f"{path}: {field}: ")
