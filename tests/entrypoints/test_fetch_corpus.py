"""``rag-fetch-corpus``: exit codes, output and files, over ``httpx.MockTransport``."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx
import pytest

from rag_load_test.adapters.corpora.catalog import CORPORA
from rag_load_test.config.settings import RagSettings
from rag_load_test.entrypoints.fetch_corpus import main

QUEBEC = CORPORA["quebec-insurance"]
A25_PATH, AMF_PATH = QUEBEC.reference_paths[0], QUEBEC.reference_paths[1]


def quebec_files(**overrides: str | int) -> dict[str, str | int]:
    """Every file of the corpus: two tiny reference files, the rest empty."""
    files: dict[str, str | int] = dict.fromkeys(QUEBEC.reference_paths, "")
    files[A25_PATH] = '{"text": "La prime.", "url": "https://a25.test"}\n'
    files[AMF_PATH] = '\n{"text": "Le sinistre."}\n'
    files[QUEBEC.questions_path] = '{"question": "Qui paie ?", "answer": "a"}\n'
    return {**files, **overrides}


def serve(files: Mapping[str, str | int]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        found = files.get(str(request.url).removeprefix(f"{QUEBEC.base_url}/"), 404)
        if isinstance(found, int):
            return httpx.Response(found)
        return httpx.Response(200, text=found)

    return httpx.MockTransport(handler)


@pytest.fixture
def quiet_settings(settings_factory: Callable[..., RagSettings]) -> RagSettings:
    return settings_factory(http_retries=0, http_backoff_s=0.0)


def test_fetch_writes_both_files_and_prints_a_summary_and_the_licence(
    tmp_path: Path, quiet_settings: RagSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "quebec"
    argv = ["quebec-insurance", "--out", str(out)]
    assert main(argv, settings=quiet_settings, transport=serve(quebec_files())) == 0

    corpus = [json.loads(line) for line in (out / "corpus.jsonl").open()]
    assert corpus == [
        {
            "id": "A-25:00001",
            "title": "A-25",
            "text": "La prime.",
            "metadata": {"source": "A-25", "url": "https://a25.test"},
        },
        {
            "id": "AMF:00002",
            "title": "AMF",
            "text": "Le sinistre.",
            "metadata": {"source": "AMF"},
        },
    ]
    assert (out / "questions.jsonl").read_text() == '{"question": "Qui paie ?"}\n'
    summary_line, licence_line = capsys.readouterr().out.splitlines()
    assert json.loads(summary_line) == {
        "event": "corpus_fetched",
        "corpus": "quebec-insurance",
        "documents": 2,
        "questions": 1,
        "corpus_file": str(out / "corpus.jsonl"),
        "questions_file": str(out / "questions.jsonl"),
    }
    assert licence_line == f"licence: {QUEBEC.licence_note}"


@pytest.mark.parametrize(
    ("override", "names"),
    [
        ({AMF_PATH: 404}, [f"{QUEBEC.base_url}/{AMF_PATH}", "HTTP 404"]),
        ({AMF_PATH: '{"text": "ok"}\n{broken\n'}, [AMF_PATH, "line 2"]),
    ],
    ids=["http-404", "malformed-line"],
)
def test_an_expected_failure_exits_1_without_a_traceback_or_files(
    tmp_path: Path,
    quiet_settings: RagSettings,
    capsys: pytest.CaptureFixture[str],
    override: dict[str, str | int],
    names: list[str],
) -> None:
    transport = serve(quebec_files(**override))
    argv = ["quebec-insurance", "--out", str(tmp_path)]
    assert main(argv, settings=quiet_settings, transport=transport) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("rag-fetch-corpus: ")
    assert all(name in captured.err for name in names)
    assert "Traceback" not in captured.err
    assert list(tmp_path.iterdir()) == []


def test_an_unknown_corpus_is_refused_by_the_parser(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        main(["bsard", "--out", str(tmp_path)])
    assert info.value.code == 2
    assert "quebec-insurance" in capsys.readouterr().err
