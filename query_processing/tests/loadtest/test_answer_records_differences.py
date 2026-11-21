"""``qp-compare-answers``: what differs is named, what cannot be compared is refused.

A baseline is recorded from the workflow app on its fake backends, then a run
is made of it by changing one thing in its file, as another deployment would
have answered.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from answer_record_helpers import (
    compare,
    record,
    use_loadtest_variables,
    workflow_client,
)
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.loadtest.answer_record import read_answer_record
from rag_components import FAKE_BACKENDS

SettingsFactory = Callable[..., WorkflowSettings]
Edit = Callable[[dict[str, Any]], None]
QUERY = 1  # the second request sent: the first question's /query


@pytest.fixture
def baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings_factory: SettingsFactory
) -> Path:
    """The fake app's record, under ``tmp_path / "baseline"``."""
    use_loadtest_variables(monkeypatch, tmp_path, {})
    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        assert record(client, tmp_path / "baseline") == 0
    return tmp_path / "baseline"


def run_edited(baseline: Path, edit: Edit) -> Path:
    """A run whose record is the baseline's with ``edit`` applied to it."""
    document = json.loads(Path(f"{baseline}_answers.json").read_text(encoding="utf-8"))
    edit(document)
    run = baseline.with_name("run")
    Path(f"{run}_answers.json").write_text(json.dumps(document), encoding="utf-8")
    return run


def compared(baseline: Path, run: Path) -> tuple[int, str]:
    """The status of comparing ``run`` with ``baseline``, and the report."""
    out = baseline.with_name("comparison.md")
    status = compare(baseline, run, out=out)
    return status, out.read_text(encoding="utf-8") if out.exists() else ""


def records_of(document: dict[str, Any]) -> list[dict[str, Any]]:
    """The vector-store records of the first question's ``/query`` answer."""
    return document["answers"][QUERY]["records"]


def swap_first_two(document: dict[str, Any]) -> None:
    found = records_of(document)
    found[0], found[1] = found[1], found[0]


def rename_first(document: dict[str, Any]) -> None:
    records_of(document)[0]["metadata"]["id"] = "another-record"


def change_text(document: dict[str, Any]) -> None:
    records_of(document)[0]["text_sha256"] = "0" * 64


def move_score(by: float) -> Edit:
    def edit(document: dict[str, Any]) -> None:
        records_of(document)[0]["rerank_score"] += by

    return edit


@pytest.mark.parametrize(
    ("edit", "field"),
    [
        (swap_first_two, "records.metadata.id"),
        (rename_first, "records.metadata.id"),
        (change_text, "records[0].text_sha256"),
        (move_score(0.01), "records[0].rerank_score"),
    ],
    ids=["out-of-order", "id-changed", "text-changed", "score-beyond-tolerance"],
)
def test_a_changed_vector_store_record_makes_the_run_different_naming_request_and_field(
    baseline: Path, edit: Edit, field: str
) -> None:
    question = read_answer_record(baseline).answers[QUERY].body["question"]

    status, report = compared(baseline, run_edited(baseline, edit))

    assert status == 1
    assert "| run | 6 | 5 | 1 |" in report
    assert f"| 2 | /query | {question} | {field} |" in report


def test_a_score_moved_within_the_tolerance_is_the_same(baseline: Path) -> None:
    status, report = compared(baseline, run_edited(baseline, move_score(1e-6)))

    assert status == 0, report


def test_an_absent_score_equals_only_an_absent_one(baseline: Path) -> None:
    def drop_score(document: dict[str, Any]) -> None:
        records_of(document)[0]["rerank_score"] = None

    status, report = compared(baseline, run_edited(baseline, drop_score))

    assert status == 1
    assert "records[0].rerank_score" in report


@pytest.mark.parametrize(
    ("variables", "expected"),
    [({}, 0), ({"QP_LOADTEST_ANSWER_ROUGE_L": "true"}, 1)],
    ids=["ignored-by-default", "rouge-l-when-set"],
)
def test_the_answer_text_is_held_with_rouge_l_only_when_the_setting_says_so(
    baseline: Path,
    monkeypatch: pytest.MonkeyPatch,
    variables: dict[str, str],
    expected: int,
) -> None:
    def reword(document: dict[str, Any]) -> None:
        document["answers"][QUERY]["answer"] = "another answer"

    run = run_edited(baseline, reword)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    status, report = compared(baseline, run)

    assert status == expected
    assert ("| answer (ROUGE-L 0.000) |" in report) is bool(expected)


CONDITION = [{"field": "year", "op": "eq", "value": 2024}]


@pytest.mark.parametrize(
    ("ours", "theirs", "expected"),
    [(None, [], 0), (None, CONDITION, 1), ([], CONDITION, 1)],
    ids=["absent-against-empty", "absent-against-a-condition", "empty-against-one"],
)
def test_an_absent_metadata_filter_equals_only_an_empty_one(
    baseline: Path, ours: list[Any] | None, theirs: list[Any], expected: int
) -> None:
    # An app from before the field existed records no filter: no condition.
    def set_filter(value: list[Any] | None) -> Edit:
        def edit(document: dict[str, Any]) -> None:
            document["answers"][QUERY]["metadata_filter"] = value

        return edit

    run = run_edited(baseline, set_filter(theirs))
    record_file = Path(f"{baseline}_answers.json")
    document = json.loads(record_file.read_text(encoding="utf-8"))
    set_filter(ours)(document)
    record_file.write_text(json.dumps(document), encoding="utf-8")

    status, report = compared(baseline, run)

    assert status == expected, report
    assert ("| metadata_filter |" in report) is bool(expected)


def test_records_of_different_requests_are_refused_naming_the_first_that_differs(
    baseline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def ask_otherwise(document: dict[str, Any]) -> None:
        document["answers"][2]["body"]["question"] = "another question"

    status, report = compared(baseline, run_edited(baseline, ask_otherwise))

    assert status == 2
    assert report == ""  # refused before any comparison
    assert "run: request 3 is /retrieve {'question': 'another question'}" in (
        capsys.readouterr().err
    )


def test_records_of_a_different_number_of_requests_are_refused(
    baseline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, _ = compared(baseline, run_edited(baseline, lambda d: d["answers"].pop()))

    assert status == 2
    assert "run holds 5 requests, baseline 6" in capsys.readouterr().err


def test_a_record_of_the_wrong_shape_names_its_file_its_field_and_the_shape(
    baseline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def break_status(document: dict[str, Any]) -> None:
        document["answers"][0]["status"] = "ok"

    run = run_edited(baseline, break_status)

    status, _ = compared(baseline, run)

    error = capsys.readouterr().err
    assert status == 2
    assert f"{run}_answers.json: answers.0.status:" in error
    assert 'expected an answer record {"seen": [' in error


def test_an_id_beside_the_metadata_in_the_new_form_is_refused_as_any_unknown_key(
    baseline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Only a list of the old form has its id beside its metadata; under
    # "records" one is a key the record does not know.
    def put_an_id_beside(document: dict[str, Any]) -> None:
        records_of(document)[0]["id"] = "beside"

    run = run_edited(baseline, put_an_id_beside)

    status, _ = compared(baseline, run)

    error = capsys.readouterr().err
    assert status == 2
    assert f"{run}_answers.json: answers.1.records.0.id: Extra inputs" in error


def test_two_runs_of_one_name_are_refused_not_one_dropped(
    baseline: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A run is named by its prefix's last component: two such would be one.
    other = tmp_path / "elsewhere"
    other.mkdir()
    copied = Path(f"{baseline}_answers.json").read_text(encoding="utf-8")
    Path(f"{other / 'baseline'}_answers.json").write_text(copied, encoding="utf-8")

    status = compare(baseline, other / "baseline", out=tmp_path / "c.md")

    assert status == 2
    assert "two runs named 'baseline'" in capsys.readouterr().err
