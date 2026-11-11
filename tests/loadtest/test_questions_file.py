"""``RAG_LOADTEST_QUESTIONS_FILE``: the users ask a corpus's own questions."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from rag_load_test.loadtest.request_plans import (
    build_request_pool,
    load_questions,
    plan_embeddings_request,
    plan_rerank_request,
    read_loadtest_config,
)
from repo_paths import REPO_ROOT

QUESTIONS_FILE = "RAG_LOADTEST_QUESTIONS_FILE"


def written(tmp_path: Path, text: str, name: str = "questions.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_the_config_reads_the_questions_file_and_leaves_it_unset_by_default(
    tmp_path: Path,
) -> None:
    assert read_loadtest_config({}).questions_file is None
    assert read_loadtest_config({QUESTIONS_FILE: ""}).questions_file is None
    path = str(tmp_path / "q.jsonl")
    assert read_loadtest_config({QUESTIONS_FILE: path}).questions_file == Path(path)


def test_json_lines_give_their_question_and_skip_blank_lines(tmp_path: Path) -> None:
    path = written(
        tmp_path, '{"question": "Qui paie ?"}\n\n{"question": "Et après ?", "x": 1}\n'
    )
    assert load_questions(path) == ("Qui paie ?", "Et après ?")


def test_plain_text_gives_one_question_per_line(tmp_path: Path) -> None:
    path = written(
        tmp_path, "Qui paie ?\n\n  Quand déclarer un sinistre ?  \n", "q.txt"
    )
    assert load_questions(path) == ("Qui paie ?", "Quand déclarer un sinistre ?")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", r"questions\.jsonl.*no question"),
        ("\n  \n", r"questions\.jsonl.*no question"),
        (
            '{"question": "ok"}\n{"answer": "no question"}\n',
            r"questions\.jsonl: line 2",
        ),
        ('{"question": "ok"}\n{"question": ""}\n', r"questions\.jsonl: line 2"),
        ('{"question": "ok"}\nplain line\n', r"questions\.jsonl: line 2"),
    ],
    ids=["empty", "blank", "no-question-key", "empty-question", "not-json"],
)
def test_a_bad_questions_file_names_the_file_and_the_line(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_questions(written(tmp_path, text))


def test_a_missing_questions_file_names_it(tmp_path: Path) -> None:
    missing = tmp_path / "absent.jsonl"
    with pytest.raises(ValueError, match=rf"{QUESTIONS_FILE}.*absent\.jsonl"):
        load_questions(missing)


def test_the_pool_takes_every_question_of_the_file_whatever_the_count(
    tmp_path: Path,
) -> None:
    path = written(tmp_path, "Qui paie ?\nQuand ?\nOù ?\n")
    pool = build_request_pool(1, candidates=4, top_n=2, questions_file=path)
    assert pool.questions == ("Qui paie ?", "Quand ?", "Où ?")
    assert len(pool.rerank_documents) == 4  # still the synthetic candidates
    rng = random.Random(3)
    assert plan_rerank_request("r", pool, rng).payload["query"] in pool.questions
    assert plan_embeddings_request("e", pool, rng).payload["input"][0] in pool.questions


def test_the_locustfile_asks_the_questions_of_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = written(tmp_path, '{"question": "Qui paie ?"}\n')
    monkeypatch.setenv(QUESTIONS_FILE, str(path))
    monkeypatch.setenv("RAG_LOADTEST_QUESTIONS", "40")  # does not apply to a file
    spec = importlib.util.spec_from_file_location(
        "locustfile_with_questions_file", REPO_ROOT / "locustfile.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.POOL.questions == ("Qui paie ?",)
