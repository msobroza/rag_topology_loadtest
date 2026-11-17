"""``QP_LOADTEST_QUESTIONS_FILE``: the users ask a corpus's own questions."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from loadtest_settings_helpers import settings_from
from query_processing.loadtest.request_plans import (
    build_request_pool,
    load_questions,
    plan_embeddings_request,
    plan_rerank_request,
)
from repo_paths import TOOLKIT_ROOT

QUESTIONS_FILE = "QP_LOADTEST_QUESTIONS_FILE"


def written(tmp_path: Path, text: str, name: str = "questions.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_the_settings_read_the_questions_file_and_leave_it_unset_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert settings_from(monkeypatch, {}).questions_file is None
    assert settings_from(monkeypatch, {QUESTIONS_FILE: ""}).questions_file is None
    path = str(tmp_path / "q.jsonl")
    settings = settings_from(monkeypatch, {QUESTIONS_FILE: path})
    assert settings.questions_file == Path(path)


def test_json_lines_give_their_question_and_skip_blank_lines(tmp_path: Path) -> None:
    path = written(
        tmp_path, '{"question": "Qui paie ?"}\n\n{"question": "Et après ?", "x": 1}\n'
    )
    assert load_questions(path) == (
        {"question": "Qui paie ?"},
        {"question": "Et après ?"},
    )


def test_a_plain_text_line_is_a_question_alone(tmp_path: Path) -> None:
    path = written(
        tmp_path, "Qui paie ?\n\n  Quand déclarer un sinistre ?  \n", "q.txt"
    )
    assert load_questions(path) == (
        {"question": "Qui paie ?"},
        {"question": "Quand déclarer un sinistre ?"},
    )


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


def test_a_line_can_carry_a_history_and_a_filter(tmp_path: Path) -> None:
    body = {
        "question": "and b?",
        "conversational_history": [
            {"role": "user", "content": "a?"},
            {"role": "assistant", "content": "A."},
        ],
        "metadata_filter": [{"attribute": "topic", "comparator": "eq", "value": "x"}],
    }
    assert load_questions(written(tmp_path, json.dumps(body))) == (body,)


def test_another_key_of_a_line_is_left_out(tmp_path: Path) -> None:
    line = json.dumps({"question": "q?", "answer": "evaluation material"})
    assert load_questions(written(tmp_path, line)) == ({"question": "q?"},)


def test_the_file_decides_the_mix_of_first_messages_and_follow_ups(
    tmp_path: Path,
) -> None:
    first = {"question": "a?"}
    follow_up = {
        "question": "and b?",
        "conversational_history": [{"role": "user", "content": "a?"}],
    }
    text = "\n".join(json.dumps(body) for body in (first, follow_up, first))
    assert load_questions(written(tmp_path, text)) == (first, follow_up, first)


@pytest.mark.parametrize(
    "extra",
    [
        {"conversational_history": "a?"},
        {"conversational_history": [{"role": "system", "content": "a?"}]},
        {"conversational_history": [{"role": "user", "content": ""}]},
        {"metadata_filter": {"attribute": "topic"}},
        {"metadata_filter": [{"attribute": "topic", "comparator": "like"}]},
    ],
    ids=[
        "history-not-a-list",
        "bad-role",
        "empty-message",
        "filter-not-a-list",
        "bad-condition",
    ],
)
def test_a_line_whose_history_or_filter_is_not_a_body_names_the_line(
    tmp_path: Path, extra: dict[str, object]
) -> None:
    text = '{"question": "ok"}\n' + json.dumps({"question": "q?", **extra})
    with pytest.raises(ValueError, match=r"questions\.jsonl: line 2"):
        load_questions(written(tmp_path, text))


def test_a_bad_history_message_is_named_by_its_place(tmp_path: Path) -> None:
    history = [{"role": "user", "content": "a?"}, {"role": "system", "content": "b"}]
    line = json.dumps({"question": "q?", "conversational_history": history})
    with pytest.raises(ValueError, match=r"at conversational_history\.1\.role"):
        load_questions(written(tmp_path, line))


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ('{"question": "ok"}\n' + json.dumps({"question": "x" * 2001}), "q.jsonl"),
        ("ok\n" + "x" * 2001, "q.txt"),
    ],
    ids=["json-line", "plain-text-line"],
)
def test_a_question_longer_than_the_app_takes_is_refused_naming_the_line(
    tmp_path: Path, text: str, name: str
) -> None:
    # Regression of the review: the loader missed the body's limit on a
    # question's length, so the line gave 422s during the run.
    with pytest.raises(
        ValueError, match=r"line 2: .* at question: String should have at most 2000"
    ):
        load_questions(written(tmp_path, text, name))


def test_a_question_of_the_longest_length_the_app_takes_is_read(
    tmp_path: Path,
) -> None:
    path = written(tmp_path, json.dumps({"question": "x" * 2000}))
    assert load_questions(path) == ({"question": "x" * 2000},)


def test_the_question_of_a_follow_up_is_stripped(tmp_path: Path) -> None:
    history = [{"role": "user", "content": "a?"}]
    line = json.dumps({"question": "  and b? ", "conversational_history": history})
    expected = {"question": "and b?", "conversational_history": history}
    assert load_questions(written(tmp_path, line)) == (expected,)


def test_a_missing_questions_file_names_it(tmp_path: Path) -> None:
    missing = tmp_path / "absent.jsonl"
    with pytest.raises(ValueError, match=rf"{QUESTIONS_FILE}.*absent\.jsonl"):
        load_questions(missing)


def test_the_pool_takes_every_question_of_the_file_whatever_the_count(
    tmp_path: Path,
) -> None:
    path = written(tmp_path, "Qui paie ?\nQuand ?\nOù ?\n")
    pool = build_request_pool(1, candidates=4, top_n=2, questions_file=path)
    questions = ("Qui paie ?", "Quand ?", "Où ?")
    assert pool.questions == tuple({"question": q} for q in questions)
    assert len(pool.rerank_documents) == 4  # still the synthetic candidates
    rng = random.Random(3)
    assert plan_rerank_request("r", pool, rng).payload["query"] in questions
    assert plan_embeddings_request("e", pool, rng).payload["input"][0] in questions


def test_the_locustfile_asks_the_questions_of_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = written(tmp_path, '{"question": "Qui paie ?"}\n')
    monkeypatch.setenv(QUESTIONS_FILE, str(path))
    monkeypatch.setenv("QP_LOADTEST_QUESTIONS", "40")  # does not apply to a file
    spec = importlib.util.spec_from_file_location(
        "locustfile_with_questions_file", TOOLKIT_ROOT / "locustfile.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.POOL.questions == ({"question": "Qui paie ?"},)
