"""Calibration of a CPU-bound stage: time each candidate here, keep the fastest.

A CPU-bound stage plugs in by subclassing ``BaseCalibratedStage``: it says how
to load its adapter for a thread count, how to give a loaded one another batch
size, and how to serve one request of the usual size. ``calibrate`` does the
rest, the same way for every stage:

- the first thread count (the thread budget's share) is loaded and every batch
  size is timed on it, the whole request first;
- each later thread count is loaded and timed with the best batch size so far
  only: batch size first, then threads, because a load is the expensive part;
- a trial is one untimed warm-up request, then ``rounds`` timed ones, and counts
  as the fewest seconds of them;
- once ``budget_s`` has passed no new trial starts, the first always runs.

Only the adapter of the fastest thread count so far is kept, so two models are
never held longer than one comparison. One ``stage_calibrated`` JSON line
reports the choice and every trial.

Example:
    >>> reranker = await calibrate(
    ...     RerankStage(model, pool, settings),
    ...     threads=(2, 1),
    ...     batch_sizes=(20, 10, 5, 2, 1),
    ...     limits=CalibrationLimits(budget_s=90.0, rounds=1),
    ... )
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from ...domain.calibration import StageTuning, TimedTrial, fastest
from ..json_log import log_json

T = TypeVar("T")


class BaseCalibratedStage(ABC, Generic[T]):
    """What a CPU-bound stage gives calibration; ``T`` is the stage's adapter.

    Example:
        >>> class RerankStage(BaseCalibratedStage[CrossEncoderReranker]):
        ...     stage = "rerank"
        ...     async def loaded(self, threads): ...
        ...     def rebatched(self, adapter, batch_size): ...
        ...     async def serve_sample(self, adapter): ...
    """

    stage: ClassVar[str]  # the stage's name as the timings report it: embed, rerank

    @abstractmethod
    async def loaded(self, threads: int) -> T:
        """Return the stage's adapter with its model loaded for ``threads`` threads.

        Args:
            threads: Native threads one call of the model may use.

        Returns:
            The adapter, at any batch size: ``rebatched`` sets it.
        """

    @abstractmethod
    def rebatched(self, adapter: T, batch_size: int) -> T:
        """Return ``adapter``'s loaded model at another batch size.

        Args:
            adapter: An adapter ``loaded`` returned.
            batch_size: Texts (or passages) per model batch.

        Returns:
            An adapter sharing ``adapter``'s model.
        """

    @abstractmethod
    async def serve_sample(self, adapter: T) -> None:
        """Serve one request of the usual size through ``adapter``.

        Args:
            adapter: The adapter under trial.
        """


@dataclass(frozen=True)
class CalibrationLimits:
    """How long calibration may search, and how often it times each candidate."""

    budget_s: float  # no new trial starts once this much time has passed
    rounds: int  # timed requests per trial, after one warm-up


async def calibrate(
    stage: BaseCalibratedStage[T],
    *,
    threads: Sequence[int],
    batch_sizes: Sequence[int],
    limits: CalibrationLimits,
    clock: Callable[[], float] = time.perf_counter,
) -> T:
    """Return the stage's adapter under the fastest of the candidate tunings.

    With one candidate of each parameter (both fixed by the settings), the model
    loads once and nothing is timed or logged.

    Args:
        stage: The CPU-bound stage to calibrate.
        threads: Thread counts to try, the thread budget's share first.
        batch_sizes: Batch sizes to try, the whole request first.
        limits: The time budget and the timed rounds per trial.
        clock: Seconds, as ``time.perf_counter`` gives them.

    Returns:
        The adapter of the fastest thread count, at the fastest batch size.

    Example:
        >>> embedder = await calibrate(
        ...     EmbedStage(model, pool, settings), threads=(4, 2, 1),
        ...     batch_sizes=(64,), limits=CalibrationLimits(90.0, 1))
    """
    if len(threads) == 1 and len(batch_sizes) == 1:
        return stage.rebatched(await stage.loaded(threads[0]), batch_sizes[0])
    started = clock()
    first = await stage.loaded(threads[0])
    search = _Search(stage, first, limits, clock, started)
    del first  # from here on the search alone holds a model
    return await search.run(threads, batch_sizes)


class _Search(Generic[T]):
    """One calibration: its trials so far, and the one adapter it keeps.

    Args:
        stage: The stage under calibration.
        first: Its adapter loaded for the first thread count.
        limits: The time budget and the timed rounds per trial.
        clock: Seconds, as ``time.perf_counter`` gives them.
        started: When calibration started, on ``clock``.
    """

    def __init__(
        self,
        stage: BaseCalibratedStage[T],
        first: T,
        limits: CalibrationLimits,
        clock: Callable[[], float],
        started: float,
    ) -> None:
        self._stage, self._limits, self._clock = stage, limits, clock
        self._kept = first  # the adapter of the fastest thread count so far
        self._started = started
        self._trials: list[TimedTrial] = []
        self._cut_short = False

    async def run(self, threads: Sequence[int], batch_sizes: Sequence[int]) -> T:
        for batch_size in batch_sizes:
            if not self._may_start():
                break
            await self._time(self._kept, StageTuning(threads[0], batch_size))
        for count in threads[1:]:
            if not self._may_start():
                break
            await self._try_threads(count)
        return self._finish()

    async def _try_threads(self, threads: int) -> None:
        # The candidate is dropped on return unless it became the fastest.
        best = fastest(self._trials)
        candidate = await self._stage.loaded(threads)
        await self._time(candidate, StageTuning(threads, best.batch_size))
        if fastest(self._trials).threads == threads:
            self._kept = candidate

    async def _time(self, adapter: T, tuning: StageTuning) -> None:
        sized = self._stage.rebatched(adapter, tuning.batch_size)
        await self._stage.serve_sample(sized)  # warm-up, not counted
        rounds = [await self._timed(sized) for _ in range(self._limits.rounds)]
        self._trials.append(TimedTrial(tuning, min(rounds)))

    async def _timed(self, adapter: T) -> float:
        start = self._clock()
        await self._stage.serve_sample(adapter)
        return self._clock() - start

    def _may_start(self) -> bool:
        # The first trial always runs, so a tiny budget keeps today's tuning.
        if not self._trials:
            return True
        if self._clock() - self._started < self._limits.budget_s:
            return True
        self._cut_short = True
        return False

    def _finish(self) -> T:
        chosen = fastest(self._trials)
        self._log(chosen)
        return self._stage.rebatched(self._kept, chosen.batch_size)

    def _log(self, chosen: StageTuning) -> None:
        log_json(
            event="stage_calibrated",
            stage=self._stage.stage,
            chosen={"threads": chosen.threads, "batch_size": chosen.batch_size},
            trials=[_trial_fields(trial) for trial in self._trials],
            seconds=round(self._clock() - self._started, 3),
            budget_s=self._limits.budget_s,
            cut_short=self._cut_short,
        )


def _trial_fields(trial: TimedTrial) -> dict[str, float]:
    tuning = trial.tuning
    return {
        "threads": tuning.threads,
        "batch_size": tuning.batch_size,
        "seconds": round(trial.seconds, 3),
    }
