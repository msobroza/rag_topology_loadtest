"""Calibration of a stage behind a model server: one server start per candidate.

A model server such as OVMS takes its tuning only when it starts, and its
stream count only pays when several requests are in flight. So a stage served
by one plugs in by subclassing ``BaseServedStage``: it says how to have its
server started and answering under a tuning, how to send the sample's requests
all at once, and which settings pin a tuning. ``calibrate_served`` does the rest:

- every stream count is timed on the first thread count, one stream first: it
  is what the export runs, and what a spent budget keeps;
- each later thread count that is at least the best stream count is timed with
  that stream count (a stream needs a thread);
- a trial's seconds are the time until all ``requests`` are answered; the
  warm-up, rounds and budget are ``TimedTrials``'s;
- a trial still running when the budget is spent is abandoned, its server
  stopped, and the search ends there, cut short: the fastest completed trial
  wins, or the first candidate, untimed, when none completed. A slow candidate
  never fails the search.

Example:
    >>> report = await calibrate_served(
    ...     OvmsRerankStage(...), streams=(1, 2), threads=(2, 1), batch_size=20,
    ...     requests=4, limits=CalibrationLimits(budget_s=90.0, rounds=1))
    >>> report.chosen
    StageTuning(threads=2, batch_size=20, streams=2)
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import ClassVar

from ...domain.runtime.calibration import CalibrationReport, StageTuning
from .timed_trials import CalibrationLimits, TimedTrials, untimed_report


class BaseServedStage(ABC):
    """What a stage behind a model server gives calibration.

    Example:
        >>> class OvmsRerankStage(BaseServedStage):
        ...     stage = "rerank"
        ...     def serving(self, tuning): ...
        ...     async def serve_sample(self): ...
        ...     def pinned_settings(self, tuning): ...
    """

    stage: ClassVar[str]  # the stage's name as the timings report it: embed, rerank

    @abstractmethod
    def serving(self, tuning: StageTuning) -> AbstractAsyncContextManager[None]:
        """Return a context that has the server started under ``tuning``.

        Args:
            tuning: The streams and threads the server runs the model with.

        Returns:
            An async context manager: on entry the server answers, on exit it
            is stopped.
        """

    @abstractmethod
    async def serve_sample(self) -> None:
        """Send the sample's requests at once to the server started last.

        Returns when every one is answered.
        """

    @abstractmethod
    def pinned_settings(self, tuning: StageTuning) -> dict[str, int]:
        """Return the environment variables that pin ``tuning`` for this server.

        Args:
            tuning: The tuning to pin, usually a calibration's choice.

        Returns:
            Variable name to value, as ``.env`` or ``deploy.yaml`` takes them.
        """


async def calibrate_served(  # noqa: PLR0913 - every candidate list is a keyword
    stage: BaseServedStage,
    *,
    streams: Sequence[int],
    threads: Sequence[int],
    batch_size: int,
    requests: int,
    limits: CalibrationLimits,
    clock: Callable[[], float] = time.perf_counter,
) -> CalibrationReport:
    """Return the report of the fastest stream and thread counts for ``stage``.

    With one candidate of each (both fixed by the settings), nothing is started
    or timed; the report then holds that tuning and no trial.

    Args:
        stage: The served stage to calibrate.
        streams: Stream counts to try, 1 first.
        threads: Thread counts to try, the largest first.
        batch_size: The request's size, reported in the tuning.
        requests: Requests in flight while a candidate is timed.
        limits: The time budget and the timed rounds per trial.
        clock: Seconds, as ``time.perf_counter`` gives them.

    Returns:
        The choice and every completed trial, logged as one ``stage_calibrated``
        line; the first candidate, untimed and cut short, when the budget was
        spent before any trial completed.

    Raises:
        ValueError: If a stream count exceeds the first thread count.

    Example:
        >>> (await calibrate_served(stage, streams=(1,), threads=(2,),
        ...                         batch_size=1, requests=4, limits=limits)).trials
        ()
    """
    if max(streams) > threads[0]:
        raise ValueError(
            f"stream counts {tuple(streams)} must not exceed the first thread "
            f"count {threads[0]}: a stream needs a thread"
        )
    if len(streams) == 1 and len(threads) == 1:
        tuning = StageTuning(threads[0], batch_size, streams[0])
        return untimed_report(stage.stage, tuning, limits, requests=requests)
    trials = TimedTrials(stage.stage, limits, clock, requests=requests)
    on_all_threads = [StageTuning(threads[0], batch_size, count) for count in streams]
    if await _time_each(stage, trials, on_all_threads):
        best = trials.fastest().streams
        fewer_threads = [
            StageTuning(count, batch_size, best)
            for count in threads[1:]
            if count >= best
        ]
        await _time_each(stage, trials, fewer_threads)
    return trials.finish(untimed=on_all_threads[0])


async def _time_each(
    stage: BaseServedStage, trials: TimedTrials, tunings: Sequence[StageTuning]
) -> bool:
    # True when every tuning was timed; False once the budget stopped the search.
    for tuning in tunings:
        if not trials.may_start():
            return False
        if not await _time_within_budget(stage, trials, tuning):
            return False
    return True


async def _time_within_budget(
    stage: BaseServedStage, trials: TimedTrials, tuning: StageTuning
) -> bool:
    # True when the trial completed; False when the budget ran out during it.
    # GCP sitting of 2026-11-15: 16 requests on one OVMS stream were answered
    # one after the other, about 2 s each; a slow candidate must end the search,
    # not fail the App. One server start per candidate: the start is not timed.
    async with stage.serving(tuning):
        trial = trials.time_trial(tuning, stage.serve_sample)
        try:
            await asyncio.wait_for(trial, trials.seconds_left())
        except TimeoutError:  # the server is stopped as the context ends
            trials.mark_cut_short()
            return False
    return True
