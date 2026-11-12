"""The target record: the pair a response reports, and the record written and read."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from rag_load_test.loadtest.target_record import (
    TargetRecord,
    TargetSighting,
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
        ({RUN_LABEL: "mono-a"}, ("", "mono-a")),
        # Neither header: not the workflow app answering (a proxy's error page, a
        # model server, a connection error), so there is nothing to count.
        ({"Content-Type": "text/html"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_target_pair(
    headers: dict[str, str] | None, expected: tuple[str, str] | None
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
