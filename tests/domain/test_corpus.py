"""Tests for ``domain.corpus``: the synthetic handbook, its questions and chunking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_load_test.adapters.fakes.fake_models import tokenize
from rag_load_test.domain.corpus import (
    TOPICS,
    SourceDocument,
    chunk,
    synthetic_corpus,
    synthetic_questions,
)

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
    assert own > 0
    assert max(overlap.values()) < own


# --- chunk ---


def test_chunk_word_windows_overlap_and_ids() -> None:
    passages = chunk(_doc(250), chunk_words=100, overlap_words=20)
    assert [p.id for p in passages] == ["doc-0007:0", "doc-0007:1", "doc-0007:2"]
    words = [p.text.split() for p in passages]
    assert [len(w) for w in words] == [100, 100, 90]
    assert words[1][:20] == words[0][-20:]
    assert words[0][0] == "w0"
    assert words[-1][-1] == "w249"


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
