"""Named fakes for calibration: a clock that moves only when told, and two stages.

``FakeCalibratedStage`` (in-process) records every load, rebatch and served
sample; ``FakeServedStage`` (behind a model server) every server start, stop and
served sample. Their requests "take" the seconds a test gives per tuning:
serving one advances the ``FakeClock`` by the next of that tuning's durations,
so the search measures exactly what the test scripted.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

from query_processing.domain.calibration import StageTuning
from query_processing.infrastructure.calibration.served_stage_calibration import (
    BaseServedStage,
)
from query_processing.infrastructure.local_models.stage_calibration import (
    BaseCalibratedStage,
)


class FakeClock:
    """A clock that reads ``now`` and moves only through ``advance``."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeLoadedModel:
    """What one load gives: a model fixed to its thread count."""

    def __init__(self, threads: int) -> None:
        self.threads = threads


@dataclass(frozen=True)
class FakeStageAdapter:
    """An adapter: a loaded model and the batch size it calls it with."""

    model: FakeLoadedModel
    batch_size: int

    @property
    def tuning(self) -> StageTuning:
        return StageTuning(threads=self.model.threads, batch_size=self.batch_size)


class FakeCalibratedStage(BaseCalibratedStage[FakeStageAdapter]):
    """A CPU-bound stage whose requests take the seconds scripted per tuning.

    Args:
        clock: Advanced by each load (``load_s``) and each served request.
        seconds: Per tuning, the durations of its successive requests, the
            warm-up first; past the end, the last one repeats.
        default_s: The duration of a request under a tuning ``seconds`` omits.
        load_s: The duration of one load.
    """

    stage = "fake"

    def __init__(
        self,
        clock: FakeClock,
        seconds: Mapping[StageTuning, Sequence[float]] | None = None,
        *,
        default_s: float = 1.0,
        load_s: float = 0.0,
    ) -> None:
        self._clock, self._seconds = clock, dict(seconds or {})
        self._default_s, self._load_s = default_s, load_s
        self.loads: list[int] = []
        self.models_alive_at_load: list[int] = []  # other models held at each load
        self.rebatches: list[StageTuning] = []
        self.served: list[StageTuning] = []
        self._alive: weakref.WeakSet[FakeLoadedModel] = weakref.WeakSet()

    async def loaded(self, threads: int) -> FakeStageAdapter:
        self.models_alive_at_load.append(len(self._alive))
        self.loads.append(threads)
        self._clock.advance(self._load_s)
        model = FakeLoadedModel(threads)
        self._alive.add(model)
        return FakeStageAdapter(model, batch_size=0)

    def rebatched(self, adapter: FakeStageAdapter, batch_size: int) -> FakeStageAdapter:
        rebatched = FakeStageAdapter(adapter.model, batch_size)
        self.rebatches.append(rebatched.tuning)
        return rebatched

    async def serve_sample(self, adapter: FakeStageAdapter) -> None:
        tuning = adapter.tuning
        durations = self._seconds.get(tuning, [self._default_s])
        nth = self.served.count(tuning)
        self.served.append(tuning)
        self._clock.advance(durations[min(nth, len(durations) - 1)])

    def pinned_settings(self, tuning: StageTuning) -> dict[str, int]:
        return {"FAKE_THREADS": tuning.threads, "FAKE_BATCH_SIZE": tuning.batch_size}


class FakeServedStage(BaseServedStage):
    """A stage behind a model server whose sample takes the seconds scripted per tuning.

    Args:
        clock: Advanced by each server start (``start_s``) and each served sample.
        seconds: Per tuning, the durations of its successive samples, the
            warm-up first; past the end, the last one repeats.
        default_s: The duration of a sample under a tuning ``seconds`` omits.
        start_s: The duration of one server start.
        hangs: Tunings whose samples never end: only cancelling them stops them.
    """

    stage = "fake-served"

    def __init__(
        self,
        clock: FakeClock,
        seconds: Mapping[StageTuning, Sequence[float]] | None = None,
        *,
        default_s: float = 1.0,
        start_s: float = 0.0,
        hangs: Collection[StageTuning] = (),
    ) -> None:
        self._clock, self._seconds = clock, dict(seconds or {})
        self._default_s, self._start_s = default_s, start_s
        self._hangs = frozenset(hangs)
        self.started: list[StageTuning] = []
        self.stopped: list[StageTuning] = []
        self.served: list[StageTuning] = []
        self._running: StageTuning | None = None

    @asynccontextmanager
    async def serving(self, tuning: StageTuning) -> AsyncIterator[None]:
        assert self._running is None, "two servers at once"
        self.started.append(tuning)
        self._clock.advance(self._start_s)
        self._running = tuning
        try:
            yield
        finally:
            self._running = None
            self.stopped.append(tuning)

    async def serve_sample(self) -> None:
        tuning = self._running
        assert tuning is not None, "a sample served with no server running"
        durations = self._seconds.get(tuning, [self._default_s])
        nth = self.served.count(tuning)
        self.served.append(tuning)
        if tuning in self._hangs:
            await asyncio.Event().wait()  # never set
        self._clock.advance(durations[min(nth, len(durations) - 1)])

    def pinned_settings(self, tuning: StageTuning) -> dict[str, int]:
        return {"FAKE_STREAMS": tuning.streams, "FAKE_THREADS": tuning.threads}
