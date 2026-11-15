"""``CompositionRoot.calibration``: one component calibrated on demand, every parameter searched.

The models are ``FakeSentenceTransformersModule``'s, on the real clock: which
candidates are tried is deterministic, which one wins is not, so the tests
assert the trials and that the pinned settings name the choice.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from rag_load_test.composition.calibrated_stages import ComponentCalibration
from rag_load_test.composition.composition_root import CompositionRoot
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.roles import ModelServerPart
from sentence_transformers_fakes import FakeSentenceTransformersModule

SettingsFactory = Callable[..., RagSettings]


@pytest.fixture
def fake_weights(monkeypatch: pytest.MonkeyPatch) -> FakeSentenceTransformersModule:
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def four_cpu_settings(
    settings_factory: SettingsFactory, **overrides: object
) -> RagSettings:
    """torch, 4 CPUs, 1 worker, every Backend fake: the share is 4 threads."""
    return settings_factory(
        local_model_runtime="torch",
        model_cpus=4,
        model_workers=1,
        embedding_backend="fake",
        rerank_backend="fake",
        vector_store_backend="fake",
        generation_backend="fake",
        **overrides,
    )


async def calibration_of(
    part: ModelServerPart, settings: RagSettings
) -> ComponentCalibration:
    async with CompositionRoot(settings) as composition:
        return await composition.calibration(part)


def tried(calibration: ComponentCalibration) -> list[tuple[int, int]]:
    trials = calibration.report.trials
    return [(trial.tuning.threads, trial.tuning.batch_size) for trial in trials]


async def test_the_reranker_is_searched_even_when_the_settings_pin_it(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    settings = four_cpu_settings(
        settings_factory, rerank_threads=1, rerank_batch_size=7
    )
    calibration = await calibration_of("reranker", settings)

    trials = tried(calibration)
    assert trials[:5] == [(4, 20), (4, 10), (4, 5), (4, 2), (4, 1)]
    assert [threads for threads, _ in trials[5:]] == [2, 1]
    chosen = calibration.report.chosen
    assert calibration.pinned == {
        "RAG_RERANK_THREADS": chosen.threads,
        "RAG_RERANK_BATCH_SIZE": chosen.batch_size,
    }
    assert (calibration.cpus, calibration.runtime) == (4.0, "torch")


async def test_the_embedder_searches_its_threads_at_the_configured_batch_size(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    settings = four_cpu_settings(
        settings_factory, embedding_threads=1, embedding_batch_size=16
    )
    calibration = await calibration_of("embedder", settings)

    assert tried(calibration) == [(4, 16), (2, 16), (1, 16)]
    assert calibration.report.stage == "embed"
    assert calibration.pinned == {
        "RAG_EMBEDDING_THREADS": calibration.report.chosen.threads
    }


@pytest.mark.parametrize(
    ("part", "role"), [("embedder", "embedder"), ("reranker", "reranker")]
)
async def test_only_the_part_asked_for_loads(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    part: ModelServerPart,
    role: str,
) -> None:
    await calibration_of(part, four_cpu_settings(settings_factory))

    assert fake_weights.loaded
    assert {loaded.split(":")[0] for loaded in fake_weights.loaded} == {role}


async def test_the_budget_and_rounds_come_from_the_settings(
    settings_factory: SettingsFactory, fake_weights: FakeSentenceTransformersModule
) -> None:
    settings = four_cpu_settings(
        settings_factory, calibration_budget_s=1e-9, calibration_rounds=3
    )
    calibration = await calibration_of("reranker", settings)

    report = calibration.report
    assert tried(calibration) == [(4, 20)]  # the first trial always runs
    assert (report.budget_s, report.cut_short) == (1e-9, True)
    assert fake_weights.batch_sizes == [("predict", 20)] * 4  # warm-up, 3 rounds
