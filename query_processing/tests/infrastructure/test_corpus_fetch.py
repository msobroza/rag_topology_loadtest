"""The corpus adapters: the table, the conversion, the download and the atomic write.

Every byte comes from ``httpx.MockTransport`` and a few hand-written lines: no
network, and nothing of a real corpus in this repository.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import httpx
import pytest

from rag_load_test.infrastructure.corpora.catalog import CORPORA, CorpusDefinition
from rag_load_test.infrastructure.corpora.conversion import (
    CorpusFormatError,
    corpus_questions,
    reference_documents,
)
from rag_load_test.infrastructure.corpora.download import (
    CorpusDownloader,
    CorpusFetchError,
)
from rag_load_test.infrastructure.corpora.jsonl_files import write_jsonl_atomically
from rag_load_test.infrastructure.http.http_policy import HttpPolicy

BASE = "https://corpus.test/raw"
TINY = CorpusDefinition(
    name="tiny",
    base_url=BASE,
    reference_paths=("References/V1/A-25.jsonl", "References/V1/AMF.jsonl"),
    questions_path="QnA/V1/questions-answers.jsonl",
    licence_note="research use only",
)
A25 = '{"text": "La prime est due.", "url": "https://a25.test/1"}\n\n{"text": "Sans url."}\n'
AMF = '{"text": "Un sinistre.", "url": null}\n'
QNA = '{"question": "Qui paie ?", "answer": "L\'assureur.", "criteria": "x"}\n'


class FakeSleep:
    """Records the backoff waits instead of sleeping."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def serve(files: Mapping[str, str | int]) -> httpx.MockTransport:
    """Answer each ``BASE``-relative path with its text, or the status it maps to."""

    def handler(request: httpx.Request) -> httpx.Response:
        found = files.get(str(request.url).removeprefix(f"{BASE}/"), 404)
        if isinstance(found, int):
            return httpx.Response(found, text="nope")
        return httpx.Response(200, text=found)

    return httpx.MockTransport(handler)


def downloader(
    transport: httpx.BaseTransport, retries: int = 2, sleep: FakeSleep | None = None
) -> CorpusDownloader:
    policy = HttpPolicy(timeout_s=5.0, retries=retries, backoff_s=0.5)
    return CorpusDownloader(
        httpx.Client(transport=transport), policy, sleep or FakeSleep()
    )


# --- the table ---


def test_quebec_insurance_lists_its_seven_reference_files_and_its_questions() -> None:
    corpus = CORPORA["quebec-insurance"]
    stems = [Path(p).stem for p in corpus.reference_paths]
    assert stems == [
        "A-25",
        "AMF",
        "CHAD",
        "GAA",
        "code_securite_routiere",
        "educaloi",
        "infoassurances",
    ]
    assert all(p.startswith("References/V1/") for p in corpus.reference_paths)
    assert corpus.questions_path == "QnA/V1/questions-answers.jsonl"
    assert corpus.url("QnA/V1/q.jsonl").startswith(
        "https://raw.githubusercontent.com/GRAAL-Research/quebec-insurance-rag-corpora/main/"
    )
    assert "Beauchemin et al., NLLP 2024" in corpus.licence_note


# --- conversion ---


def test_reference_lines_become_documents_with_stem_ids_and_optional_url() -> None:
    docs = reference_documents("References/V1/A-25.jsonl", A25)
    assert [d.model_dump() for d in docs] == [
        {
            "id": "A-25:00001",
            "title": "A-25",
            "text": "La prime est due.",
            "metadata": {"source": "A-25", "url": "https://a25.test/1"},
        },
        {
            "id": "A-25:00003",  # the file's line number: the blank line 2 is skipped
            "title": "A-25",
            "text": "Sans url.",
            "metadata": {"source": "A-25"},
        },
    ]


def test_a_null_url_is_left_out() -> None:
    (doc,) = reference_documents("References/V1/AMF.jsonl", AMF)
    assert doc.metadata == {"source": "AMF"}


def test_questions_keep_only_the_question() -> None:
    text = QNA + "\n" + '{"question": "Et le constat ?", "answer": "a"}\n'
    assert corpus_questions("QnA/V1/q.jsonl", text) == ["Qui paie ?", "Et le constat ?"]


@pytest.mark.parametrize(
    ("convert", "text", "line"),
    [
        (reference_documents, '{"text": "ok"}\n{"text": "cut\n', 2),
        (reference_documents, '\n["not", "an object"]\n', 2),
        (reference_documents, '{"url": "https://x.test"}\n', 1),
        (corpus_questions, '{"question": "ok"}\n\n{"answer": "no question"}\n', 3),
    ],
    ids=["broken-json", "not-an-object", "no-text", "no-question"],
)
def test_a_malformed_line_names_the_file_and_the_line(
    convert: Callable[[str, str], object], text: str, line: int
) -> None:
    with pytest.raises(CorpusFormatError) as info:
        convert("References/V1/A-25.jsonl", text)
    assert (info.value.source, info.value.lineno) == ("References/V1/A-25.jsonl", line)
    assert str(info.value).startswith(f"References/V1/A-25.jsonl: line {line}: ")


# --- download ---


def test_fetch_converts_every_reference_file_and_the_questions() -> None:
    files = {**dict.fromkeys(TINY.reference_paths, ""), TINY.questions_path: QNA}
    files[TINY.reference_paths[0]] = A25
    files[TINY.reference_paths[1]] = AMF
    fetched = downloader(serve(files)).fetch(TINY)
    assert [d.id for d in fetched.documents] == [
        "A-25:00001",
        "A-25:00003",
        "AMF:00001",
    ]
    assert fetched.questions == ["Qui paie ?"]


def test_a_5xx_is_retried_with_backoff_then_succeeds() -> None:
    answers = iter([httpx.Response(503), httpx.Response(200, text="body")])
    sleep = FakeSleep()
    fetch = downloader(httpx.MockTransport(lambda _r: next(answers)), sleep=sleep)
    assert fetch.get_text(f"{BASE}/a.jsonl") == "body"
    assert sleep.waits == [0.5]


def test_a_transport_error_is_retried_until_the_retries_run_out() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    sleep = FakeSleep()
    with pytest.raises(CorpusFetchError) as info:
        downloader(httpx.MockTransport(refuse), sleep=sleep).get_text(f"{BASE}/a")
    assert info.value.url == f"{BASE}/a"
    assert "refused" in str(info.value)
    assert sleep.waits == [0.5, 1.0]


def test_a_404_fails_at_once_naming_the_url_and_the_status() -> None:
    sleep = FakeSleep()
    with pytest.raises(CorpusFetchError) as info:
        downloader(serve({}), sleep=sleep).get_text(f"{BASE}/References/V1/X.jsonl")
    assert str(info.value) == f"{BASE}/References/V1/X.jsonl: HTTP 404"
    assert sleep.waits == []


# --- the atomic write ---


def test_write_jsonl_atomically_writes_one_object_per_line(tmp_path: Path) -> None:
    out = tmp_path / "questions.jsonl"
    out.write_text("old content\n")  # a re-run overwrites
    count = write_jsonl_atomically(out, [{"question": "Où ?"}, {"question": "Quand ?"}])
    assert count == 2
    assert out.read_text(encoding="utf-8") == (
        '{"question": "Où ?"}\n{"question": "Quand ?"}\n'
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["questions.jsonl"]


def test_an_interrupted_write_leaves_the_old_file_and_no_temporary(
    tmp_path: Path,
) -> None:
    out = tmp_path / "corpus.jsonl"
    out.write_text("previous run\n")

    def interrupted() -> Iterator[dict[str, object]]:
        yield {"id": "a"}
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        write_jsonl_atomically(out, interrupted())
    assert out.read_text() == "previous run\n"
    assert [p.name for p in tmp_path.iterdir()] == ["corpus.jsonl"]
