"""``domain.runtime.calibration``: the candidates, the fastest trial and the sample request."""

from __future__ import annotations

import pytest

from query_processing.domain.runtime.calibration import (
    CalibrationReport,
    StageTuning,
    TimedTrial,
    calibration_sample,
    fastest,
    halvings,
)

# --- halvings ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("top", "expected"),
    [
        (20, (20, 10, 5, 2, 1)),
        (8, (8, 4, 2, 1)),
        (3, (3, 1)),
        (1, (1,)),
    ],
)
def test_halvings_go_from_top_down_to_one(top: int, expected: tuple[int, ...]) -> None:
    assert halvings(top) == expected


@pytest.mark.parametrize("top", [0, -3])
def test_halvings_refuse_a_top_below_one(top: int) -> None:
    with pytest.raises(ValueError, match=f"top={top}"):
        halvings(top)


# --- fastest -------------------------------------------------------------------------


def trial(threads: int, batch_size: int, seconds: float) -> TimedTrial:
    return TimedTrial(StageTuning(threads=threads, batch_size=batch_size), seconds)


def test_fastest_is_the_tuning_with_the_fewest_seconds() -> None:
    trials = [trial(2, 20, 5.5), trial(2, 2, 3.4), trial(1, 2, 4.0)]

    assert fastest(trials) == StageTuning(threads=2, batch_size=2)


def test_a_tie_keeps_the_tuning_tried_first() -> None:
    trials = [trial(2, 20, 3.0), trial(2, 10, 3.0), trial(1, 10, 3.0)]

    assert fastest(trials) == StageTuning(threads=2, batch_size=20)


def test_fastest_of_no_trial_is_an_error() -> None:
    with pytest.raises(ValueError, match="no trial"):
        fastest([])


# --- the sample request --------------------------------------------------------------


def test_the_sample_is_deterministic() -> None:
    assert calibration_sample(20) == calibration_sample(20)


@pytest.mark.parametrize("records", [1, 2, 20, 50])
def test_the_sample_has_the_records_asked(records: int) -> None:
    sample = calibration_sample(records)

    assert len(sample.records) == records
    assert sample.question.endswith("?")


def test_the_sample_records_spread_from_short_to_long() -> None:
    lengths = sorted(len(p) for p in calibration_sample(20).records)

    assert 100 <= lengths[0] <= 200
    assert 700 <= lengths[-1] <= 800
    assert len(set(lengths)) == 20  # spread, not a few sizes repeated


def test_the_sample_records_are_not_in_length_order() -> None:
    # Retrieval returns candidates in score order, not length order.
    lengths = [len(p) for p in calibration_sample(20).records]

    assert lengths not in (sorted(lengths), sorted(lengths, reverse=True))


def test_the_sample_refuses_no_record() -> None:
    with pytest.raises(ValueError, match="records=0"):
        calibration_sample(0)


# --- streams and requests --------------------------------------------------------------


def test_an_in_process_tuning_runs_one_stream() -> None:
    assert StageTuning(threads=2, batch_size=20).streams == 1
    assert StageTuning(2, 20) != StageTuning(2, 20, streams=2)


def test_a_report_serves_one_request_at_once_unless_told() -> None:
    chosen = StageTuning(2, 20, streams=2)
    trials = (TimedTrial(chosen, 2.41),)
    report = CalibrationReport("rerank", chosen, trials, 71.0, 90.0, False)

    assert report.requests == 1
    assert CalibrationReport(
        "rerank", chosen, trials, 71.0, 90.0, False, requests=4
    ).chosen_seconds == pytest.approx(2.41)
