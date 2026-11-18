"""Calibration of a CPU-bound stage: time each candidate here, keep the fastest.

A CPU-bound stage plugs in by subclassing ``BaseCalibratedStage``: it says how
to load its adapter for a thread count, how to give a loaded one another batch
size, how to serve one request of the usual size, and which settings pin a
tuning. ``calibrate`` does the rest, the same way for every stage:

- the first thread count (the thread budget's share) is loaded and every batch
  size is timed on it, the whole request first;
- each later thread count is loaded and timed with the best batch size so far
  only: batch size first, then threads, because a load is the expensive part;
- a trial is one untimed warm-up request, then ``rounds`` timed ones, and counts
  as the fewest seconds of them;
- once ``budget_s`` has passed no new trial starts, the first always runs.

The last two rules, and the report, are ``TimedTrials``'s
(``infrastructure/calibration/timed_trials.py``), shared with the calibration
of a model server.

Only the adapter of the fastest thread count so far is kept, so two models are
never held longer than one comparison. The result is the adapter and a
``CalibrationReport`` of the choice and every trial; one ``stage_calibrated``
JSON line logs that report.

Example:
    >>> calibrated = await calibrate(
    ...     CalibratedRerankStage(model, pool, passages=20, top_n=10),
    ...     threads=(2, 1),
    ...     batch_sizes=(20, 10, 5, 2, 1),
    ...     limits=CalibrationLimits(budget_s=90.0, rounds=1),
    ... )
    >>> reranker = calibrated.adapter
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from ...domain.runtime.calibration import CalibrationReport, StageTuning
from ..calibration.timed_trials import CalibrationLimits, TimedTrials, untimed_report

T = TypeVar("T")


class BaseCalibratedStage(ABC, Generic[T]):
    """What a CPU-bound stage gives calibration; ``T`` is the stage's adapter.

    Example:
        >>> class CalibratedRerankStage(BaseCalibratedStage[CrossEncoderReranker]):
        ...     stage = "rerank"
        ...     async def loaded(self, threads): ...
        ...     def rebatched(self, adapter, batch_size): ...
        ...     async def serve_sample(self, adapter): ...
        ...     def pinned_settings(self, tuning): ...
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

    @abstractmethod
    def pinned_settings(self, tuning: StageTuning) -> dict[str, int]:
        """Return the settings of the component's ``local`` block that pin ``tuning``.

        Only the parameters calibration searches for this stage are named, so
        an App started with them calibrates nothing.

        Args:
            tuning: The tuning to pin, usually a calibration's choice.

        Returns:
            Setting name in the ``local`` block to value, such as
            ``{"threads": 2}``.
        """


@dataclass(frozen=True)
class Calibrated(Generic[T]):
    """A calibrated stage's adapter, and the report of how it was chosen."""

    adapter: T  # under the chosen tuning
    report: CalibrationReport


async def calibrate(
    stage: BaseCalibratedStage[T],
    *,
    threads: Sequence[int],
    batch_sizes: Sequence[int],
    limits: CalibrationLimits,
    clock: Callable[[], float] = time.perf_counter,
) -> Calibrated[T]:
    """Return the stage's adapter under the fastest of the candidate tunings.

    With one candidate of each parameter (both fixed by the settings), the model
    loads once and nothing is timed or logged; the report then holds that tuning,
    no trial and 0 seconds.

    Args:
        stage: The CPU-bound stage to calibrate.
        threads: Thread counts to try, the thread budget's share first.
        batch_sizes: Batch sizes to try, the whole request first.
        limits: The time budget and the timed rounds per trial.
        clock: Seconds, as ``time.perf_counter`` gives them.

    Returns:
        The adapter of the fastest thread count, at the fastest batch size, and
        the report of the search.

    Example:
        >>> calibrated = await calibrate(
        ...     CalibratedEmbedStage(
        ...         model, pool, query_prefix="query: ", normalize_embeddings=False),
        ...     threads=(4, 2, 1), batch_sizes=(64,),
        ...     limits=CalibrationLimits(90.0, 1))
        >>> calibrated.report.chosen
        StageTuning(threads=2, batch_size=64)
    """
    if len(threads) == 1 and len(batch_sizes) == 1:
        return await _fixed(stage, StageTuning(threads[0], batch_sizes[0]), limits)
    trials = TimedTrials(stage.stage, limits, clock)  # the budget counts loads too
    first = await stage.loaded(threads[0])
    search = _Search(stage, first, trials)
    del first  # from here on the search alone holds a model
    return await search.run(threads, batch_sizes)


async def _fixed(
    stage: BaseCalibratedStage[T], tuning: StageTuning, limits: CalibrationLimits
) -> Calibrated[T]:
    adapter = stage.rebatched(await stage.loaded(tuning.threads), tuning.batch_size)
    return Calibrated(adapter, untimed_report(stage.stage, tuning, limits))


class _Search(Generic[T]):
    """One calibration: its trials so far, and the one adapter it keeps.

    Args:
        stage: The stage under calibration.
        first: Its adapter loaded for the first thread count.
        trials: The trials, started before ``first`` loaded.
    """

    def __init__(
        self, stage: BaseCalibratedStage[T], first: T, trials: TimedTrials
    ) -> None:
        self._stage, self._trials = stage, trials
        self._kept = first  # the adapter of the fastest thread count so far

    async def run(
        self, threads: Sequence[int], batch_sizes: Sequence[int]
    ) -> Calibrated[T]:
        for batch_size in batch_sizes:
            if not self._trials.may_start():
                break
            await self._time(self._kept, StageTuning(threads[0], batch_size))
        for count in threads[1:]:
            if not self._trials.may_start():
                break
            await self._try_threads(count)
        report = self._trials.finish()
        adapter = self._stage.rebatched(self._kept, report.chosen.batch_size)
        return Calibrated(adapter, report)

    async def _try_threads(self, threads: int) -> None:
        # The candidate is dropped on return unless it became the fastest.
        best = self._trials.fastest()
        candidate = await self._stage.loaded(threads)
        await self._time(candidate, StageTuning(threads, best.batch_size))
        if self._trials.fastest().threads == threads:
            self._kept = candidate

    async def _time(self, adapter: T, tuning: StageTuning) -> None:
        sized = self._stage.rebatched(adapter, tuning.batch_size)
        await self._trials.time_trial(tuning, lambda: self._stage.serve_sample(sized))
