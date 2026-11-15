"""``calibrate_served``: one server start per candidate, streams first, then threads."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from calibration_fakes import FakeClock, FakeServedStage
from rag_load_test.domain.calibration import CalibrationReport, StageTuning
from rag_load_test.infrastructure.calibration.served_stage_calibration import (
    calibrate_served,
)
from rag_load_test.infrastructure.calibration.timed_trials import CalibrationLimits

LIMITS = CalibrationLimits(budget_s=1000.0, rounds=1)


def tuning(threads: int, streams: int) -> StageTuning:
    return StageTuning(threads=threads, batch_size=20, streams=streams)


async def search(
    stage: FakeServedStage,
    clock: FakeClock,
    streams: Sequence[int],
    threads: Sequence[int],
    limits: CalibrationLimits = LIMITS,
) -> CalibrationReport:
    return await calibrate_served(
        stage,
        streams=streams,
        threads=threads,
        batch_size=20,
        requests=4,
        limits=limits,
        clock=clock,
    )


async def test_every_stream_count_is_timed_on_the_first_thread_count() -> None:
    clock = FakeClock()
    stage = FakeServedStage(clock)
    await search(stage, clock, streams=(1, 2, 4), threads=(4,))

    assert stage.started == [tuning(4, 1), tuning(4, 2), tuning(4, 4)]
    assert stage.stopped == stage.started  # each server stopped after its trial
    assert stage.served == [t for t in stage.started for _ in range(2)]


async def test_later_thread_counts_are_timed_with_the_best_stream_count() -> None:
    clock = FakeClock()
    seconds = {tuning(4, 2): [1.0], tuning(2, 2): [0.5]}
    stage = FakeServedStage(clock, seconds, default_s=3.0)
    report = await search(stage, clock, streams=(1, 2, 4), threads=(4, 2, 1))

    # 1 thread is below the best stream count: a stream needs a thread.
    assert stage.started[3:] == [tuning(2, 2)]
    assert report.chosen == tuning(2, 2)


async def test_a_trial_is_the_time_until_every_request_is_answered() -> None:
    clock = FakeClock()
    stage = FakeServedStage(clock, {tuning(2, 1): [9.0, 2.5]}, start_s=30.0)
    report = await search(stage, clock, streams=(1,), threads=(2, 1))

    assert report.trials[0].seconds == 2.5  # the start and the warm-up are not
    assert report.requests == 4


async def test_both_fixed_starts_nothing_and_reports_the_tuning() -> None:
    clock = FakeClock()
    stage = FakeServedStage(clock)
    report = await search(stage, clock, streams=(2,), threads=(4,))

    assert stage.started == []
    assert (report.chosen, report.trials, report.requests) == (tuning(4, 2), (), 4)


async def test_a_spent_budget_keeps_one_stream() -> None:
    clock = FakeClock()
    stage = FakeServedStage(clock, default_s=10.0)
    limits = CalibrationLimits(budget_s=1.0, rounds=1)
    report = await search(stage, clock, (1, 2, 4), (4, 2, 1), limits)

    assert stage.started == [tuning(4, 1)]
    assert (report.chosen, report.cut_short) == (tuning(4, 1), True)


async def test_a_stream_count_above_the_first_thread_count_is_refused() -> None:
    clock = FakeClock()
    with pytest.raises(ValueError, match=r"\(1, 8\).*first thread count 4"):
        await search(FakeServedStage(clock), clock, streams=(1, 8), threads=(4, 2))


async def test_the_server_is_stopped_when_a_sample_fails() -> None:
    clock = FakeClock()

    class FailingStage(FakeServedStage):
        async def serve_sample(self) -> None:
            raise OSError("connection reset")

    stage = FailingStage(clock)
    with pytest.raises(OSError, match="connection reset"):
        await search(stage, clock, streams=(1, 2), threads=(2,))

    assert stage.stopped == [tuning(2, 1)]
