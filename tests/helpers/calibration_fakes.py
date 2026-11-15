"""Named fakes for calibration: a clock that moves only when told, and a stage.

``FakeCalibratedStage`` records every load, rebatch and served sample. Its
requests "take" the seconds a test gives per tuning: serving one advances the
``FakeClock`` by the next of that tuning's durations, so ``calibrate`` measures
exactly what the test scripted.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag_load_test.domain.calibration import StageTuning
from rag_load_test.infrastructure.local_models.stage_calibration import (
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
