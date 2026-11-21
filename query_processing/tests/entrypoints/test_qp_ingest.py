"""Tests for ``entrypoints.qp_ingest``: JSONL loading and the ``qp-ingest`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from query_processing.adapters.vector_store.fake.in_memory_vector_store import (
    InMemoryVectorStore,
)
from query_processing.composition.composition_root import CompositionRoot
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.retrieval.corpus import (
    TOPICS,
    SourceDocument,
)
from query_processing.entrypoints.qp_ingest import load_jsonl, main

SLUGS = list(TOPICS)


def _doc(words: int, **metadata: str | int) -> SourceDocument:
    """Text ``w0 w1 ... w<words-1>`` so chunk boundaries are checkable."""
    text = " ".join(f"w{i}" for i in range(words))
    return SourceDocument(id="doc-0007", title="Handbook", text=text, metadata=metadata)


def _jsonl(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _record(**fields: object) -> str:
    return json.dumps({"id": "a", "title": "A", "text": "alpha", **fields})


# --- load_jsonl ---


def test_load_jsonl_skips_blank_lines_and_defaults_metadata(tmp_path: Path) -> None:
    line_b = json.dumps({"id": "b", "title": "B", "text": "beta"})
    lines = (_record(metadata={"k": 1}), "", "   ", line_b)
    docs = load_jsonl(_jsonl(tmp_path / "docs.jsonl", *lines))
    assert [(d.id, d.metadata) for d in docs] == [("a", {"k": 1}), ("b", {})]


@pytest.mark.parametrize(
    ("bad_line", "detail"),
    [('{"id": ', "Expecting value"), ('{"id": "b", "title": "B"}', "text")],
)
def test_load_jsonl_names_the_offending_line(
    tmp_path: Path, bad_line: str, detail: str
) -> None:
    path = _jsonl(tmp_path / "bad.jsonl", _record(), "", bad_line)
    with pytest.raises(ValueError, match="line 3") as excinfo:
        load_jsonl(path)
    assert detail in str(excinfo.value)


# --- qp-ingest CLI ---


def test_main_ingests_the_synthetic_corpus_and_prints_one_summary_line(
    fake_settings: WorkflowSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    composition = CompositionRoot(fake_settings, vector_store=InMemoryVectorStore())

    assert main(["--synthetic", "3"], composition=composition) == 0

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "event": "ingest_done",
        "documents": 3,
        "records": 3,
        "embedder": "fake-embedder",
        "store": "3 records in memory",
    }


def test_main_chunks_a_corpus_file(
    fake_settings: WorkflowSettings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    short = json.dumps({"id": "b", "title": "B", "text": "short"})
    path = _jsonl(tmp_path / "docs.jsonl", _record(text=" ".join(["w"] * 30)), short)
    opts = ["--chunk-words", "20", "--overlap-words", "5"]
    composition = CompositionRoot(fake_settings, vector_store=InMemoryVectorStore())

    assert main(["--corpus", str(path), *opts], composition=composition) == 0

    summary = json.loads(capsys.readouterr().out)
    assert (summary["event"], summary["documents"]) == ("ingest_done", 2)
    assert summary["records"] == 3  # 30 words at 20/5 -> 2 chunks, plus 1
    assert summary["store"] == "3 records in memory"


def test_main_refuses_the_fake_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QP_COMPONENTS__VECTOR_STORE__BACKEND", "fake")

    with pytest.raises(SystemExit, match="seeded when the app starts") as refused:
        main(["--synthetic", "1"])

    # Chroma keeps what it stores too: the refusal names no one backend.
    assert "a backend that keeps what it stores, any but 'fake'" in str(refused.value)
