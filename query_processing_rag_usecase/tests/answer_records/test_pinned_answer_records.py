"""The RAG's workflow app on fake backends answers as its pinned answer records.

The app is built from this directory's workflow file with every component on
its fake backend, recorded in process with ``qp-record-answers`` on the
questions of ``questions.jsonl`` (a first message, two follow-ups with a
conversational history, then a message with a metadata filter that repeats
its condition), each sent to ``/retrieve`` and ``/query``, and compared with
the record committed here with ``qp-compare-answers``: every answer
identical, its text exactly. The fake router chooses the first of its
routes, so one record keeps the routes in the file's order, where a follow-up
stands alone, and the other puts ``leans_on_history`` first, where it is
rewritten. The same comparison runs on every registered orchestrator, with
the records LangGraph pinned. ``make pin-answers`` runs this file with
``--pin-answers``, which makes both records again instead, on LangGraph, the
default, whatever ``QP_ORCHESTRATOR`` the shell holds (conftest's
``use_case_dir`` unsets every ``QP_`` variable).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from query_processing.testing import (
    PinnedAnswerRecord,
    check_pinned_answers,
    registered_orchestrators,
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


@pytest.mark.parametrize("name", sorted(ROUTES))
@pytest.mark.parametrize("orchestrator", registered_orchestrators())
def test_the_app_on_every_orchestrator_answers_as_the_records_langgraph_pinned(
    use_case_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    orchestrator: str,
    name: str,
) -> None:
    del use_case_dir  # requested so the settings are read from this directory
    # The records LangGraph answered before any orchestrator was added; never
    # pinned again from here, whatever --pin-answers says.
    settings = all_fake_settings(
        monkeypatch,
        QP_COMPONENTS__ROUTER__ROUTES=ROUTES[name],
        QP_ORCHESTRATOR=orchestrator,
    )

    assert settings.orchestrator == orchestrator
    with workflow_client(settings) as client:
        check_pinned_answers(
            monkeypatch, client, pinned_record(name), tmp_path, pin=False
        )


def test_the_two_records_part_at_the_follow_ups_alone() -> None:
    alone, rewritten = (
        json.loads(pinned_record(name).path.read_text(encoding="utf-8"))["answers"]
        for name in ("follow_ups_stand_alone", "follow_ups_rewritten")
    )

    # A message with no history (its /retrieve and /query) takes neither
    # route: the first, and the last; each follow-up is searched with its
    # rewrite in one record, as sent in the other.
    assert len(alone) == len(rewritten) == 8
    assert alone[:2] == rewritten[:2]
    assert alone[6:] == rewritten[6:]
    follow_ups = zip(alone[2:6], rewritten[2:6], strict=True)
    assert all(a["passages"] != r["passages"] for a, r in follow_ups)


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_a_condition_sent_twice_is_applied_once(name: str) -> None:
    answers = json.loads(pinned_record(name).path.read_text(encoding="utf-8"))[
        "answers"
    ]

    # The last request of questions.jsonl, to /retrieve and /query: a client
    # repeats its condition, the one field a reducer adds up.
    retrieved, queried = answers[6:]
    for answer in (retrieved, queried):
        (condition,) = answer["metadata_filter"]
        assert answer["body"]["metadata_filter"] == [condition, condition]
