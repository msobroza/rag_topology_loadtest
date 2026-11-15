"""``calibrate``: each candidate timed on a fake stage and a fake clock, the fastest kept."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from calibration_fakes import FakeCalibratedStage, FakeClock, FakeStageAdapter
from rag_load_test.domain.calibration import (
    CalibrationReport,
    StageTuning,
    TimedTrial,
)
from rag_load_test.infrastructure.calibration.timed_trials import CalibrationLimits
from rag_load_test.infrastructure.local_models.stage_calibration import (
    Calibrated,
    calibrate,
)

LIMITS = CalibrationLimits(budget_s=1000.0, rounds=1)


def tuning(threads: int, batch_size: int) -> StageTuning:
    return StageTuning(threads=threads, batch_size=batch_size)


async def run(
    stage: FakeCalibratedStage,
    clock: FakeClock,
    threads: Sequence[int],
    batch_sizes: Sequence[int],
    limits: CalibrationLimits = LIMITS,
) -> FakeStageAdapter:
    calibrated = await search(stage, clock, threads, batch_sizes, limits)
    return calibrated.adapter


async def search(
    stage: FakeCalibratedStage,
    clock: FakeClock,
    threads: Sequence[int],
    batch_sizes: Sequence[int],
    limits: CalibrationLimits = LIMITS,
) -> Calibrated[FakeStageAdapter]:
    return await calibrate(
        stage, threads=threads, batch_sizes=batch_sizes, limits=limits, clock=clock
    )


def calibrated_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    lines = [json.loads(r.message) for r in caplog.records]
    return [line for line in lines if line["event"] == "stage_calibrated"]


# --- both parameters fixed -----------------------------------------------------------


async def test_fixed_parameters_load_once_and_time_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    stage = FakeCalibratedStage(clock)
    with caplog.at_level("INFO", logger="rag_load_test"):
        adapter = await run(stage, clock, threads=(2,), batch_sizes=(32,))

    assert adapter.tuning == tuning(2, 32)
    assert stage.loads == [2]
    assert stage.served == []
    assert calibrated_lines(caplog) == []


# --- the search order ----------------------------------------------------------------


async def test_batch_sizes_are_searched_on_the_first_thread_count_only() -> None:
    clock = FakeClock()
    stage = FakeCalibratedStage(clock)
    await run(stage, clock, threads=(2, 1), batch_sizes=(20, 10, 5))

    batches_on_2 = [t.batch_size for t in stage.served if t.threads == 2]
    batches_on_1 = {t.batch_size for t in stage.served if t.threads == 1}
    assert batches_on_2 == [20, 20, 10, 10, 5, 5]  # warm-up, then one round, each
    assert batches_on_1 == {20}  # all tie: the first tried is the best batch size
    assert stage.loads == [2, 1]


async def test_a_later_thread_count_is_timed_with_the_best_batch_size() -> None:
    clock = FakeClock()
    seconds = {tuning(4, 5): [2.0], tuning(2, 5): [1.5]}
    stage = FakeCalibratedStage(clock, seconds, default_s=3.0)
    adapter = await run(stage, clock, threads=(4, 2, 1), batch_sizes=(20, 10, 5))

    assert [t for t in stage.served if t.threads != 4] == [tuning(2, 5)] * 2 + [
        tuning(1, 5)
    ] * 2
    assert adapter.tuning == tuning(2, 5)


async def test_the_warm_up_is_not_counted_and_a_trial_is_its_fewest_round(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    seconds = {tuning(2, 20): [100.0, 3.0, 2.0], tuning(2, 10): [0.5, 2.5, 2.5]}
    stage = FakeCalibratedStage(clock, seconds)
    limits = CalibrationLimits(budget_s=1000.0, rounds=2)
    with caplog.at_level("INFO", logger="rag_load_test"):
        adapter = await run(stage, clock, (2,), (20, 10), limits)

    [line] = calibrated_lines(caplog)
    assert [t["seconds"] for t in line["trials"]] == [2.0, 2.5]  # type: ignore[index]
    assert adapter.tuning == tuning(2, 20)  # its warm-up of 100 s did not count
    assert stage.served.count(tuning(2, 20)) == 3  # one warm-up, two rounds


# --- the budget ----------------------------------------------------------------------


async def test_the_budget_stops_new_trials_but_the_first_always_runs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    stage = FakeCalibratedStage(clock, default_s=10.0)
    limits = CalibrationLimits(budget_s=0.001, rounds=1)
    with caplog.at_level("INFO", logger="rag_load_test"):
        adapter = await run(stage, clock, (2, 1), (20, 10, 5), limits)

    assert adapter.tuning == tuning(2, 20)  # today's: the share, the whole request
    assert stage.loads == [2]
    assert stage.served == [tuning(2, 20)] * 2
    [line] = calibrated_lines(caplog)
    assert line["cut_short"] is True


async def test_the_budget_counts_from_the_start_and_stops_between_trials() -> None:
    clock = FakeClock()
    stage = FakeCalibratedStage(clock, default_s=1.0, load_s=3.0)
    limits = CalibrationLimits(budget_s=6.0, rounds=1)
    await run(stage, clock, (2, 1), (20, 10, 5), limits)

    # A load of 3 s, then 2 s per trial: 5 s after the first, 7 s after the second.
    assert stage.served == [tuning(2, 20)] * 2 + [tuning(2, 10)] * 2
    assert stage.loads == [2]


# --- what is kept --------------------------------------------------------------------


async def test_the_adapter_returned_holds_the_fastest_tuning() -> None:
    clock = FakeClock()
    seconds = {tuning(4, 10): [2.0], tuning(1, 10): [1.0]}
    stage = FakeCalibratedStage(clock, seconds, default_s=3.0)
    adapter = await run(stage, clock, threads=(4, 2, 1), batch_sizes=(20, 10))

    assert adapter.tuning == tuning(1, 10)
    assert adapter.model.threads == 1  # the model loaded for that thread count
    assert stage.rebatches[-1] == tuning(1, 10)


async def test_at_most_one_other_model_is_held_while_a_candidate_loads() -> None:
    clock = FakeClock()
    seconds = {tuning(8, 20): [1.0], tuning(2, 20): [0.5]}
    stage = FakeCalibratedStage(clock, seconds, default_s=3.0)
    await run(stage, clock, threads=(8, 4, 2, 1), batch_sizes=(20,))

    assert stage.loads == [8, 4, 2, 1]
    assert stage.models_alive_at_load == [0, 1, 1, 1]


# --- the log line --------------------------------------------------------------------


async def test_one_log_line_names_the_stage_the_choice_and_every_trial(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    seconds = {tuning(2, 20): [5.4912], tuning(2, 10): [3.3801], tuning(1, 10): [4.0]}
    stage = FakeCalibratedStage(clock, seconds, load_s=1.0)
    with caplog.at_level("INFO", logger="rag_load_test"):
        await run(stage, clock, threads=(2, 1), batch_sizes=(20, 10))

    assert calibrated_lines(caplog) == [
        {
            "event": "stage_calibrated",
            "stage": "fake",
            "chosen": {"threads": 2, "batch_size": 10, "streams": 1},
            "trials": [
                {"threads": 2, "batch_size": 20, "streams": 1, "seconds": 5.491},
                {"threads": 2, "batch_size": 10, "streams": 1, "seconds": 3.38},
                {"threads": 1, "batch_size": 10, "streams": 1, "seconds": 4.0},
            ],
            "requests": 1,
            # two loads of 1 s, and two requests (warm-up, round) per trial
            "seconds": 27.743,
            "budget_s": 1000.0,
            "cut_short": False,
        }
    ]


# --- the report ----------------------------------------------------------------------


async def test_the_report_holds_the_trials_in_order_and_the_choice() -> None:
    clock = FakeClock()
    seconds = {tuning(2, 20): [5.0], tuning(2, 10): [3.0], tuning(1, 10): [4.0]}
    stage = FakeCalibratedStage(clock, seconds, load_s=1.0)
    calibrated = await search(stage, clock, threads=(2, 1), batch_sizes=(20, 10))

    assert calibrated.report == CalibrationReport(
        stage="fake",
        chosen=tuning(2, 10),
        trials=(
            TimedTrial(tuning(2, 20), 5.0),
            TimedTrial(tuning(2, 10), 3.0),
            TimedTrial(tuning(1, 10), 4.0),
        ),
        seconds=26.0,  # two loads of 1 s, and two requests (warm-up, round) per trial
        budget_s=1000.0,
        cut_short=False,
    )
    assert calibrated.report.chosen_seconds == 3.0
    assert calibrated.adapter.tuning == calibrated.report.chosen


async def test_the_report_says_when_the_budget_cut_the_search_short() -> None:
    clock = FakeClock()
    stage = FakeCalibratedStage(clock, default_s=10.0)
    limits = CalibrationLimits(budget_s=5.0, rounds=1)
    calibrated = await search(stage, clock, (2, 1), (20, 10, 5), limits)

    report = calibrated.report
    assert report.cut_short is True
    assert [trial.tuning for trial in report.trials] == [tuning(2, 20)]
    assert (report.seconds, report.budget_s) == (20.0, 5.0)


async def test_fixed_parameters_report_the_fixed_tuning_and_no_trial() -> None:
    clock = FakeClock()
    calibrated = await search(FakeCalibratedStage(clock), clock, (3,), (7,))

    assert calibrated.report == CalibrationReport(
        stage="fake",
        chosen=tuning(3, 7),
        trials=(),
        seconds=0.0,
        budget_s=1000.0,
        cut_short=False,
    )
    assert calibrated.report.chosen_seconds is None
