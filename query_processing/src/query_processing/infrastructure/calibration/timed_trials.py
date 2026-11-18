"""What every calibration search shares: the trials, the budget, the report.

An in-process stage (``local_models/stage_calibration.py``) and a model server
(``served_stage_calibration.py``) search different parameters in different
orders, but time a candidate the same way, stop on the same budget rule and
report the same shape. ``TimedTrials`` owns that part:

- a trial is one untimed warm-up, then ``rounds`` timed calls, and counts as the
  fewest seconds of them;
- the first trial always runs; once ``budget_s`` has passed since the search
  started no new trial starts, and the report says it was cut short;
- a search that can cancel a trial (a served stage) waits for it at most
  ``seconds_left``, and on expiry calls ``mark_cut_short``; when no trial
  completed, ``finish`` keeps the tuning its caller names;
- the report is a ``CalibrationReport``, logged as one ``stage_calibrated`` JSON
  line.

Example:
    >>> trials = TimedTrials("rerank", CalibrationLimits(90.0, 1))
    >>> if trials.may_start():
    ...     await trials.time_trial(StageTuning(2, 20), serve_one_request)
    >>> report = trials.finish()
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ...domain.runtime.calibration import (
    CalibrationReport,
    StageTuning,
    TimedTrial,
    fastest,
)
from ..json_log import log_json

Serve = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class CalibrationLimits:
    """How long calibration may search, and how often it times each candidate."""

    # No new trial starts once this much time has passed; a served stage's trial
    # still running then is abandoned.
    budget_s: float
    rounds: int  # timed calls per trial, after one warm-up


class TimedTrials:
    """One calibration search's trials so far, timed on ``clock``.

    The search starts when this object is built: the budget counts from then,
    loads and server starts included.

    Args:
        stage: The stage's name as the timings report it: embed, rerank.
        limits: The time budget and the timed rounds per trial.
        clock: Seconds, as ``time.perf_counter`` gives them.
        requests: Requests one timed call serves at once, for the report.

    Example:
        >>> trials = TimedTrials("embed", CalibrationLimits(90.0, 1), requests=4)
    """

    def __init__(
        self,
        stage: str,
        limits: CalibrationLimits,
        clock: Callable[[], float] = time.perf_counter,
        *,
        requests: int = 1,
    ) -> None:
        self._stage, self._limits, self._clock = stage, limits, clock
        self._requests = requests
        self._started = clock()
        self._trials: list[TimedTrial] = []
        self._cut_short = False

    def may_start(self) -> bool:
        """Say whether a new trial may start, marking the search cut short if not.

        Returns:
            True for the first trial (so a tiny budget keeps today's tuning) and
            while the budget lasts; False once it has passed.

        Example:
            >>> trials.may_start()
            True
        """
        if not self._trials:
            return True
        if self._clock() - self._started < self._limits.budget_s:
            return True
        self._cut_short = True
        return False

    def seconds_left(self) -> float:
        """Return the seconds left of the budget, counted from the search's start.

        Returns:
            ``budget_s`` minus the seconds spent so far; 0.0 once it is spent.

        Example:
            >>> trials.seconds_left()
            61.5
        """
        return max(0.0, self._limits.budget_s - (self._clock() - self._started))

    def mark_cut_short(self) -> None:
        """Record that the budget stopped the search: a trial was abandoned.

        Example:
            >>> trials.mark_cut_short()
        """
        self._cut_short = True

    async def time_trial(self, tuning: StageTuning, serve: Serve) -> None:
        """Time ``serve`` under ``tuning``: one warm-up, then the timed rounds.

        Args:
            tuning: The candidate being timed, recorded with its seconds.
            serve: One call of the stage under that candidate.

        Example:
            >>> await trials.time_trial(StageTuning(2, 20), serve_one_request)
        """
        await serve()  # warm-up, not counted
        rounds = [await self._timed(serve) for _ in range(self._limits.rounds)]
        self._trials.append(TimedTrial(tuning, min(rounds)))

    def fastest(self) -> StageTuning:
        """Return the tuning of the fastest trial so far.

        Returns:
            The fastest tuning; on a tie, the one tried first.

        Raises:
            ValueError: If no trial has run yet.

        Example:
            >>> trials.fastest()
            StageTuning(threads=2, batch_size=20, streams=1)
        """
        return fastest(self._trials)

    def finish(self, untimed: StageTuning | None = None) -> CalibrationReport:
        """Return the report of the search, and log it as one JSON line.

        Args:
            untimed: The tuning to keep when no trial completed (the first was
                abandoned); ignored once one did.

        Returns:
            The choice, every completed trial in order, the seconds spent since
            the search started, and whether the budget cut it short.

        Raises:
            ValueError: If no trial completed and ``untimed`` is None.

        Example:
            >>> trials.finish(untimed=StageTuning(2, 20)).chosen
            StageTuning(threads=2, batch_size=20, streams=1)
        """
        report = CalibrationReport(
            stage=self._stage,
            chosen=self._chosen(untimed),
            trials=tuple(self._trials),
            seconds=self._clock() - self._started,
            budget_s=self._limits.budget_s,
            cut_short=self._cut_short,
            requests=self._requests,
        )
        _log(report)
        return report

    def _chosen(self, untimed: StageTuning | None) -> StageTuning:
        if self._trials or untimed is None:
            return fastest(self._trials)  # raises when there is no trial
        return untimed

    async def _timed(self, serve: Serve) -> float:
        start = self._clock()
        await serve()
        return self._clock() - start


def untimed_report(
    stage: str, tuning: StageTuning, limits: CalibrationLimits, *, requests: int = 1
) -> CalibrationReport:
    """Return the report of a calibration with one candidate: nothing was timed.

    Args:
        stage: The stage's name: embed, rerank.
        tuning: The one candidate, which the settings fixed.
        limits: The budget, reported as it was set.
        requests: Requests a trial would have served at once.

    Returns:
        A report holding ``tuning``, no trial and 0 seconds. Nothing is logged.

    Example:
        >>> untimed_report("embed", StageTuning(2, 64), limits).trials
        ()
    """
    return CalibrationReport(
        stage=stage,
        chosen=tuning,
        trials=(),
        seconds=0.0,
        budget_s=limits.budget_s,
        cut_short=False,
        requests=requests,
    )


def _log(report: CalibrationReport) -> None:
    log_json(
        event="stage_calibrated",
        stage=report.stage,
        chosen=_tuning_fields(report.chosen),
        trials=[
            {**_tuning_fields(trial.tuning), "seconds": round(trial.seconds, 3)}
            for trial in report.trials
        ],
        requests=report.requests,
        seconds=round(report.seconds, 3),
        budget_s=report.budget_s,
        cut_short=report.cut_short,
    )


def _tuning_fields(tuning: StageTuning) -> dict[str, int]:
    return {
        "threads": tuning.threads,
        "batch_size": tuning.batch_size,
        "streams": tuning.streams,
    }
