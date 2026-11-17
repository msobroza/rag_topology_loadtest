"""``qp-compare-answers``: an LLM's answers are held with ROUGE-L, never text for text.

A baseline is recorded from the workflow app on its fake backends; then the
first question's ``/query`` answer is written, in the baseline and in a run,
as each test says. The texts are French, accents included, as the answers are.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from answer_record_helpers import (
    compare,
    record,
    use_loadtest_variables,
    workflow_client,
)
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.loadtest.rouge_l import rouge_l_f_measure
from rag_components import FAKE_BACKENDS

SettingsFactory = Callable[..., WorkflowSettings]
QUERY = 1  # the second request sent: the first question's /query
ROUGE_L = {"QP_LOADTEST_ANSWER_ROUGE_L": "true"}
SAID = "L'été dernier, la Société a réduit ses émissions de 12 %."
REWORDED = "La Société a réduit ses émissions de 12 % l'été dernier."
# The library's own result on SAID and REWORDED with the tokenizer that keeps
# every letter: 8 of 11 words in sequence. Its default tokenizer, which drops
# the accented letters, gives 10/13 (0.769...).
REWORDED_SCORE = 8 / 11


@pytest.fixture
def baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings_factory: SettingsFactory
) -> Path:
    """The fake app's record, under ``tmp_path / "baseline"``."""
    use_loadtest_variables(monkeypatch, tmp_path, {})
    with workflow_client(settings_factory(**FAKE_BACKENDS)) as client:
        assert record(client, tmp_path / "baseline") == 0
    return tmp_path / "baseline"


def answered(prefix: Path, text: str | None, into: Path | None = None) -> Path:
    """Write ``text`` as the first ``/query`` answer of ``prefix``'s record."""
    target = into or prefix
    document = json.loads(Path(f"{prefix}_answers.json").read_text(encoding="utf-8"))
    document["answers"][QUERY]["answer"] = text
    Path(f"{target}_answers.json").write_text(json.dumps(document), encoding="utf-8")
    return target


def compared_texts(
    baseline: Path, said: str | None, reworded: str | None
) -> tuple[int, str]:
    """Compare a run answering ``reworded`` with a baseline answering ``said``."""
    run = answered(baseline, reworded, into=baseline.with_name("run"))
    answered(baseline, said)
    out = baseline.with_name("comparison.md")
    status = compare(baseline, run, out=out)
    return status, out.read_text(encoding="utf-8") if out.exists() else ""


def set_variables(monkeypatch: pytest.MonkeyPatch, variables: dict[str, str]) -> None:
    for name, value in variables.items():
        monkeypatch.setenv(name, value)


def test_the_same_text_scores_1() -> None:
    assert rouge_l_f_measure(SAID, SAID) == 1.0


def test_a_reworded_french_answer_scores_between_0_and_1_every_letter_kept() -> None:
    assert rouge_l_f_measure(SAID, REWORDED) == pytest.approx(REWORDED_SCORE, abs=1e-9)


def test_the_same_text_is_the_same_answer(
    baseline: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_variables(monkeypatch, ROUGE_L)

    status, report = compared_texts(baseline, SAID, SAID)

    assert status == 0, report
    assert "| run | 6 | 6 | 0 | 1.000 | 1.000 |" in report


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [("0.7", 0), ("0.75", 1)],
    ids=["threshold-under-the-score", "threshold-above-the-score"],
)
def test_a_reworded_answer_is_the_same_or_different_as_the_threshold_is_set(
    baseline: Path, monkeypatch: pytest.MonkeyPatch, threshold: str, expected: int
) -> None:
    set_variables(
        monkeypatch, {**ROUGE_L, "QP_LOADTEST_ANSWER_ROUGE_L_THRESHOLD": threshold}
    )

    status, report = compared_texts(baseline, SAID, REWORDED)

    assert status == expected, report
    # Lowest and mean over the three /query answers: two the same, one reworded.
    mean = (2 + REWORDED_SCORE) / 3
    assert f"| {6 - expected} | {expected} | 0.727 | {mean:.3f} |" in report
    assert (f"| answer (ROUGE-L 0.727) | {SAID} | {REWORDED} |" in report) is bool(
        expected
    )


def test_two_empty_answers_are_the_same(
    baseline: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_variables(monkeypatch, ROUGE_L)

    status, report = compared_texts(baseline, "", "")

    assert status == 0, report


@pytest.mark.parametrize(
    ("said", "reworded"),
    [("", SAID), (SAID, ""), (None, SAID), (SAID, None)],
    ids=["empty-in-baseline", "empty-in-run", "absent-in-baseline", "absent-in-run"],
)
def test_an_answer_empty_or_absent_on_one_side_only_is_different(
    baseline: Path, monkeypatch: pytest.MonkeyPatch, said: str | None, reworded: str
) -> None:
    set_variables(monkeypatch, ROUGE_L)

    status, report = compared_texts(baseline, said, reworded)

    assert status == 1
    assert "| run | 6 | 5 | 1 |" in report


def test_the_text_is_ignored_when_the_setting_is_off_and_the_report_says_so(
    baseline: Path,
) -> None:
    status, report = compared_texts(baseline, SAID, "Une tout autre réponse.")

    assert status == 0, report
    assert "| run | 6 | 6 | 0 |" in report
    assert "| run | requests | identical | different |\n" in report
    assert "The answers' text is not compared" in report


def test_without_the_library_the_command_stops_naming_the_extra(
    baseline: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A stand-in for the import: None in sys.modules makes it raise ImportError.
    monkeypatch.setitem(sys.modules, "rouge_score", None)
    monkeypatch.delitem(sys.modules, "query_processing.loadtest.rouge_l", raising=False)
    set_variables(monkeypatch, ROUGE_L)

    status, report = compared_texts(baseline, SAID, REWORDED)

    assert status == 2
    assert report == ""
    assert "pip install -e './query_processing[loadtest]'" in capsys.readouterr().err
