"""``TimedTrials``: the warm-up and rounds, the budget rule, the report and its log line."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

import pytest

from calibration_fakes import FakeClock
from query_processing.domain.runtime.calibration import (
    CalibrationReport,
    StageTuning,
    TimedTrial,
)
from query_processing.infrastructure.calibration.timed_trials import (
    CalibrationLimits,
    TimedTrials,
    untimed_report,
)


def serving(clock: FakeClock, durations: list[float]) -> Callable[[], Awaitable[None]]:
    """A serve callable that advances ``clock`` by the next duration each call."""
    calls = iter(durations)

    async def serve() -> None:
        clock.advance(next(calls))

    return serve


async def test_a_trial_skips_its_warm_up_and_keeps_its_fewest_round() -> None:
    clock = FakeClock()
    trials = TimedTrials("rerank", CalibrationLimits(1000.0, rounds=3), clock)

    await trials.time_trial(StageTuning(2, 20), serving(clock, [50.0, 3.0, 2.0, 4.0]))

    report = trials.finish()
    assert report.trials == (TimedTrial(StageTuning(2, 20), 2.0),)
    assert report.seconds == 59.0


async def test_the_first_trial_always_runs_then_the_budget_cuts_short() -> None:
    clock = FakeClock()
    trials = TimedTrials("embed", CalibrationLimits(budget_s=1.0, rounds=1), clock)

    assert trials.may_start()
    await trials.time_trial(StageTuning(2, 1), serving(clock, [5.0, 5.0]))
    assert not trials.may_start()

    assert trials.finish().cut_short is True


async def test_the_budget_counts_from_construction_loads_included() -> None:
    clock = FakeClock()
    trials = TimedTrials("embed", CalibrationLimits(budget_s=10.0, rounds=1), clock)
    clock.advance(4.0)  # a load or a server start, before the first trial
    await trials.time_trial(StageTuning(2, 1), serving(clock, [3.0, 2.0]))

    assert trials.may_start()  # 9 s spent of 10
    clock.advance(1.0)
    assert not trials.may_start()  # 10 s spent


async def test_the_report_carries_streams_and_requests_into_its_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    trials = TimedTrials("rerank", CalibrationLimits(100.0, 1), clock, requests=4)
    await trials.time_trial(StageTuning(2, 20, streams=1), serving(clock, [1, 2.5]))
    await trials.time_trial(StageTuning(2, 20, streams=2), serving(clock, [1, 1.25]))

    with caplog.at_level("INFO", logger="query_processing"):
        report = trials.finish()

    assert (report.chosen, report.requests) == (StageTuning(2, 20, streams=2), 4)
    [line] = [json.loads(record.message) for record in caplog.records]
    assert line == {
        "event": "stage_calibrated",
        "stage": "rerank",
        "chosen": {"threads": 2, "batch_size": 20, "streams": 2},
        "trials": [
            {"threads": 2, "batch_size": 20, "streams": 1, "seconds": 2.5},
            {"threads": 2, "batch_size": 20, "streams": 2, "seconds": 1.25},
        ],
        "requests": 4,
        "seconds": 5.75,
        "budget_s": 100.0,
        "cut_short": False,
    }


def test_the_seconds_left_count_down_from_the_budget_to_zero() -> None:
    clock = FakeClock()
    trials = TimedTrials("rerank", CalibrationLimits(budget_s=10.0, rounds=1), clock)

    clock.advance(4.0)
    assert trials.seconds_left() == 6.0
    clock.advance(7.0)
    assert trials.seconds_left() == 0.0


def test_with_no_trial_completed_the_report_keeps_the_tuning_given(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    trials = TimedTrials("rerank", CalibrationLimits(1.0, 1), clock, requests=16)
    clock.advance(1.5)  # a server start, then a trial abandoned
    trials.mark_cut_short()

    with caplog.at_level("INFO", logger="query_processing"):
        report = trials.finish(untimed=StageTuning(8, 20, streams=1))

    assert (report.chosen, report.trials) == (StageTuning(8, 20, streams=1), ())
    assert (report.seconds, report.cut_short) == (1.5, True)
    [line] = [json.loads(record.message) for record in caplog.records]
    assert (line["trials"], line["cut_short"]) == ([], True)


async def test_the_tuning_given_is_ignored_once_a_trial_completed() -> None:
    clock = FakeClock()
    trials = TimedTrials("rerank", CalibrationLimits(100.0, 1), clock)
    await trials.time_trial(StageTuning(2, 20, streams=2), serving(clock, [1, 1]))

    report = trials.finish(untimed=StageTuning(2, 20, streams=1))

    assert report.chosen == StageTuning(2, 20, streams=2)


def test_finishing_with_no_trial_names_the_problem() -> None:
    trials = TimedTrials("embed", CalibrationLimits(1.0, 1), FakeClock())

    with pytest.raises(ValueError, match="no trial"):
        trials.finish()


def test_an_untimed_report_holds_the_one_tuning_and_no_trial(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tuning = StageTuning(2, 20, streams=2)
    with caplog.at_level("INFO", logger="query_processing"):
        report = untimed_report(
            "rerank", tuning, CalibrationLimits(90.0, 1), requests=4
        )

    assert report == CalibrationReport(
        stage="rerank",
        chosen=tuning,
        trials=(),
        seconds=0.0,
        budget_s=90.0,
        cut_short=False,
        requests=4,
    )
    assert caplog.records == []
