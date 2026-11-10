"""Tests for ``rag_load_test.corpus``: synthetic handbook, JSONL loading, chunking, CLI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_load_test.corpus import (
    TOPICS,
    SourceDocument,
    chunk,
    ingest,
    load_jsonl,
    main,
    synthetic_corpus,
    synthetic_questions,
)
from rag_load_test.fakes import FakeEmbedder, InMemoryVectorStore, tokenize

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


# --- synthetic corpus ---


def test_synthetic_corpus_is_deterministic_per_seed() -> None:
    assert synthetic_corpus(12) == synthetic_corpus(12)
    assert synthetic_corpus(12) != synthetic_corpus(12, seed=99)


def test_synthetic_corpus_cycles_topics_and_numbers_ids() -> None:
    docs = synthetic_corpus(len(SLUGS) + 2)
    assert [d.id for d in docs[:3]] == ["doc-0000", "doc-0001", "doc-0002"]
    for i, doc in enumerate(docs):
        assert doc.metadata == {"topic": SLUGS[i % len(SLUGS)], "seq": i}


def test_synthetic_questions_count_and_cycling() -> None:
    questions = synthetic_questions(len(SLUGS) + 3)
    assert len(questions) == len(SLUGS) + 3
    for i, question in enumerate(questions):
        template = TOPICS[SLUGS[i % len(SLUGS)]].question
        assert question.startswith(template.split("{")[0])


@pytest.mark.parametrize("slug", SLUGS)
def test_question_shares_most_words_with_its_own_topic(slug: str) -> None:
    """FakeEmbedder is bag-of-words: retrieval only works if this overlap holds."""
    question = set(tokenize(TOPICS[slug].question))
    overlap = {
        s: len(question & set(tokenize(" ".join(t.sentences))))
        for s, t in TOPICS.items()
    }
    own = overlap.pop(slug)
    assert own > 0 and max(overlap.values()) < own


# --- chunk ---


def test_chunk_word_windows_overlap_and_ids() -> None:
    passages = chunk(_doc(250), chunk_words=100, overlap_words=20)
    assert [p.id for p in passages] == ["doc-0007:0", "doc-0007:1", "doc-0007:2"]
    words = [p.text.split() for p in passages]
    assert [len(w) for w in words] == [100, 100, 90]
    assert words[1][:20] == words[0][-20:]
    assert words[0][0] == "w0" and words[-1][-1] == "w249"


def test_chunk_metadata_carries_document_fields() -> None:
    (passage,) = chunk(_doc(30, topic="remote-work", seq=4))
    doc_fields = {"doc_id": "doc-0007", "title": "Handbook", "chunk": 0}
    assert passage.metadata == {"topic": "remote-work", "seq": 4, **doc_fields}


def test_chunk_empty_text_gives_no_passages() -> None:
    assert chunk(_doc(0)) == []
    assert chunk(SourceDocument(id="d", title="t", text="  \n ")) == []


@pytest.mark.parametrize("overlap", [10, 15, -1])
def test_chunk_rejects_overlap_not_below_chunk(overlap: int) -> None:
    with pytest.raises(ValueError, match=f"got {overlap}, 10"):
        chunk(_doc(30), chunk_words=10, overlap_words=overlap)


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


# --- ingest ---


class SpyEmbedder(FakeEmbedder):
    """FakeEmbedder recording the size of every ``embed`` batch."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return await super().embed(texts)


async def test_ingest_upserts_every_passage_in_batches() -> None:
    passages = [p for doc in synthetic_corpus(7) for p in chunk(doc)]
    assert len(passages) == 7  # each synthetic document fits one chunk
    embedder, store = SpyEmbedder(), InMemoryVectorStore()
    assert await ingest(passages, embedder, store, batch_size=3) == 7
    assert await store.count() == 7
    assert embedder.batch_sizes == [3, 3, 1]
    hits = await store.query(embedder.embed_text(passages[4].text), top_k=1)
    assert hits[0].id == passages[4].id


# --- rag-ingest CLI ---


def test_main_synthetic_fake_prints_one_summary_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--synthetic", "3", "--fake"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    summary = json.loads(lines[0])
    backend = {"index": "in-memory", "embedder": "fake-embedder"}
    assert summary == {"event": "ingest_done", "documents": 3, "passages": 3, **backend}


def test_main_corpus_file_fake(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    short = json.dumps({"id": "b", "title": "B", "text": "short"})
    path = _jsonl(tmp_path / "docs.jsonl", _record(text=" ".join(["w"] * 30)), short)
    opts = ["--chunk-words", "20", "--overlap-words", "5"]
    assert main(["--corpus", str(path), "--fake", *opts]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["event"], summary["documents"]) == ("ingest_done", 2)
    assert summary["passages"] == 3  # 30 words at 20/5 -> 2 chunks, plus 1
