"""The pinned answer record of the fixture workflow, and how a record is pinned.

The workflow app on the fixture workflow, every component on its fake backend,
is recorded in process and compared with the record committed in
``answer_records/``: every answer identical, on LangGraph, which answered the
record, and on every registered orchestrator, on which the timings' keys and
the tokens a record leaves out are LangGraph's. ``make pin-answers`` runs this
file with ``--pin-answers``, which makes the record again instead.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.testing import (
    PinnedAnswerRecord,
    PinnedAnswersDifferError,
    check_pinned_answers,
    registered_orchestrators,
    workflow_client,
)
from rag_components import FAKE_BACKENDS

SettingsFactory = Callable[..., WorkflowSettings]
PIN_COMMAND = "make -C query_processing pin-answers"
# Ten synthetic questions: one for each topic of the synthetic corpus.
FIXTURE_WORKFLOW = PinnedAnswerRecord(
    prefix=Path(__file__).parent / "answer_records" / "fixture_workflow",
    pin_command=PIN_COMMAND,
    variables={"QP_LOADTEST_QUESTIONS": "10"},
)


def in_tmp(tmp_path: Path) -> PinnedAnswerRecord:
    """The fixture workflow's record, pinned under ``tmp_path`` instead."""
    return PinnedAnswerRecord(
        prefix=tmp_path / "pinned" / "fixture_workflow",
        pin_command=PIN_COMMAND,
        variables=FIXTURE_WORKFLOW.variables,
    )


def test_the_app_on_fake_backends_answers_as_its_pinned_record(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pin_answers: bool,
) -> None:
    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        check_pinned_answers(
            monkeypatch, client, FIXTURE_WORKFLOW, tmp_path, pin=pin_answers
        )


@pytest.mark.parametrize("orchestrator", registered_orchestrators())
def test_the_app_on_every_orchestrator_answers_as_the_record_langgraph_pinned(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    orchestrator: str,
) -> None:
    # The record LangGraph answered before any orchestrator was added; never
    # pinned again from here.
    settings = settings_factory(**FAKE_BACKENDS, orchestrator=orchestrator)
    with workflow_client(settings) as client:
        check_pinned_answers(monkeypatch, client, FIXTURE_WORKFLOW, tmp_path, pin=False)


def measures_of(settings: WorkflowSettings) -> list[tuple[list[str], object]]:
    """What a record leaves out, for each endpoint: the timed stages, the tokens."""
    measured = []
    with workflow_client(settings) as client:
        for endpoint in ("/retrieve", "/query"):
            body = client.post(endpoint, json={"question": "How many days off?"})
            assert body.status_code == 200, body.text
            measured.append((list(body.json()["timings_ms"]), body.json()["tokens"]))
    return measured


@pytest.mark.parametrize("orchestrator", registered_orchestrators())
def test_on_every_orchestrator_the_timings_and_tokens_are_langgraph_s(
    settings_factory: SettingsFactory, orchestrator: str
) -> None:
    on_langgraph = measures_of(settings_factory(**FAKE_BACKENDS))

    on_this_one = measures_of(
        settings_factory(**FAKE_BACKENDS, orchestrator=orchestrator)
    )

    assert on_this_one == on_langgraph
    assert on_langgraph[1][0] == ["embed", "retrieve", "rerank", "generate", "total"]
    assert on_langgraph[1][1]  # the fake LLM counts its tokens


def test_a_record_not_yet_pinned_fails_naming_the_pin_command(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pinned = in_tmp(tmp_path)

    with (
        workflow_client(settings_factory(**FAKE_BACKENDS)) as client,
        pytest.raises(PinnedAnswersDifferError, match=PIN_COMMAND) as failed,
    ):
        check_pinned_answers(monkeypatch, client, pinned, tmp_path, pin=False)

    assert str(pinned.path) in str(failed.value)


def test_pinning_rewrites_the_record_file(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pinned = in_tmp(tmp_path)
    pinned.path.parent.mkdir()
    pinned.path.write_text('{"seen": [], "answers": []}', encoding="utf-8")

    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        check_pinned_answers(monkeypatch, client, pinned, tmp_path, pin=True)
        record = json.loads(pinned.path.read_text(encoding="utf-8"))
        # The record just pinned is the one the app is then held to.
        check_pinned_answers(monkeypatch, client, pinned, tmp_path, pin=False)

    assert len(record["answers"]) == 20  # ten questions, each to both endpoints
    assert [a["endpoint"] for a in record["answers"][:2]] == ["/retrieve", "/query"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("answer", "[fake-llm] another answer"), ("status", 500)],
)
def test_an_answer_that_differs_fails_naming_the_pin_command(
    settings_factory: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    pinned = in_tmp(tmp_path)
    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        check_pinned_answers(monkeypatch, client, pinned, tmp_path, pin=True)
        record = json.loads(pinned.path.read_text(encoding="utf-8"))
        record["answers"][1][field] = value  # the first /query's
        pinned.path.write_text(json.dumps(record), encoding="utf-8")

        with pytest.raises(PinnedAnswersDifferError, match=PIN_COMMAND) as failed:
            check_pinned_answers(monkeypatch, client, pinned, tmp_path, pin=False)

    # The report's row: of twenty answers, the one changed differs.
    assert "| recorded-now | 20 | 19 | 1 |" in str(failed.value)
