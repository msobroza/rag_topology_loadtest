"""The RAG's workflow app on fake backends answers as its pinned answer records.

The app is built from this directory's workflow file with every component on
its fake backend, recorded in process with ``qp-record-answers`` on the
questions of ``questions.jsonl`` (a first message, then two follow-ups with a
conversational history), each sent to ``/retrieve`` and ``/query``, and
compared with the record committed here with ``qp-compare-answers``: every
answer identical. The fake router chooses the first of its routes, so one
record keeps the routes in the file's order, where a follow-up stands alone,
and the other puts ``leans_on_history`` first, where it is rewritten.
``make pin-answers`` runs this file with ``--pin-answers``, which makes both
records again instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from query_processing.testing import (
    PinnedAnswerRecord,
    check_pinned_answers,
    workflow_client,
)

from workflow_on_fakes import all_fake_settings

HERE = Path(__file__).parent
PIN_COMMAND = "make -C query_processing_rag_usecase pin-answers"
QUESTIONS = {"QP_LOADTEST_QUESTIONS_FILE": str(HERE / "questions.jsonl")}
# Each record by its name, with the router's routes as their variable gives them.
ROUTES = {
    "follow_ups_stand_alone": '["self_contained", "leans_on_history"]',
    "follow_ups_rewritten": '["leans_on_history", "self_contained"]',
}


def pinned_record(name: str) -> PinnedAnswerRecord:
    """The record of that name, committed here."""
    return PinnedAnswerRecord(HERE / name, PIN_COMMAND, QUESTIONS)


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_the_app_on_fake_backends_answers_as_its_pinned_record(
    use_case_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pin_answers: bool,
    name: str,
) -> None:
    del use_case_dir  # requested so the settings are read from this directory
    settings = all_fake_settings(
        monkeypatch, QP_COMPONENTS__ROUTER__ROUTES=ROUTES[name]
    )

    with workflow_client(settings) as client:
        check_pinned_answers(
            monkeypatch, client, pinned_record(name), tmp_path, pin=pin_answers
        )


def test_the_two_records_part_at_the_follow_ups_alone() -> None:
    alone, rewritten = (
        json.loads(pinned_record(name).path.read_text(encoding="utf-8"))["answers"]
        for name in ("follow_ups_stand_alone", "follow_ups_rewritten")
    )

    # The first message (its /retrieve and /query) takes neither route; each
    # follow-up is searched with its rewrite in one record, as sent in the other.
    assert len(alone) == len(rewritten) == 6
    assert alone[:2] == rewritten[:2]
    follow_ups = zip(alone[2:], rewritten[2:], strict=True)
    assert all(a["passages"] != r["passages"] for a, r in follow_ups)
