"""The calibrated rerank and embed stages, and how their openers calibrate them.

The models are ``FakeSentenceTransformersModule``'s, on the real clock: which
candidates are tried is deterministic, which one wins is not, so the tests
assert the trials and that the adapter returned holds the choice.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from rag_graphs import rag_graph
from rag_load_test.adapters.embedder.local.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from rag_load_test.adapters.reranker.local.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from rag_load_test.composition.calibrated_stages import (
    CalibratedEmbedStage,
    CalibratedRerankStage,
    candidates,
)
from rag_load_test.composition.composition_root import CompositionRoot
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.calibration import StageTuning, calibration_sample
from rag_load_test.infrastructure.local_models.local_model import LocalModel
from rag_load_test.infrastructure.local_models.model_pool import model_executor
from rag_load_test.infrastructure.local_models.native_threads import NativeThreadScope
from sentence_transformers_fakes import FakeSentenceTransformersModule

SettingsFactory = Callable[..., RagSettings]
TORCH_MODEL = LocalModel("org/name", "org/name", "torch")
RERANKER = "antoinelouis/crossencoder-camembert-base-mmarcoFR"


@pytest.fixture
def fake_weights(monkeypatch: pytest.MonkeyPatch) -> FakeSentenceTransformersModule:
    module = FakeSentenceTransformersModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


@pytest.fixture
def pool() -> Any:
    executor = model_executor(1)
    yield executor
    executor.shutdown(wait=True)


def four_thread_settings(
    settings_factory: SettingsFactory, **overrides: object
) -> RagSettings:
    """torch, 4 CPUs, 1 worker: each model's share of the budget is 4 threads."""
    return settings_factory(
        local_model_runtime="torch",
        model_cpus=4,
        model_workers=1,
        vector_store_backend="fake",
        components__llm__backend="fake",
        **overrides,
    )


def calibrated_lines(caplog: pytest.LogCaptureFixture) -> dict[str, dict[str, Any]]:
    lines = [json.loads(r.message) for r in caplog.records]
    return {
        line["stage"]: line for line in lines if line["event"] == "stage_calibrated"
    }


def tried(line: dict[str, Any]) -> list[tuple[int, int]]:
    return [(trial["threads"], trial["batch_size"]) for trial in line["trials"]]


async def open_reranker(
    settings: RagSettings, caplog: pytest.LogCaptureFixture
) -> CrossEncoderReranker:
    with caplog.at_level("INFO", logger="rag_load_test"):
        async with CompositionRoot(settings) as composition:
            reranker = await composition.local_reranker("reranker")
    assert isinstance(reranker, CrossEncoderReranker)
    return reranker


async def open_embedder(
    settings: RagSettings, caplog: pytest.LogCaptureFixture
) -> SentenceTransformerEmbedder:
    with caplog.at_level("INFO", logger="rag_load_test"):
        async with CompositionRoot(settings) as composition:
            embedder = await composition.local_embedder("embedder")
    assert isinstance(embedder, SentenceTransformerEmbedder)
    return embedder


# --- the candidates ------------------------------------------------------------------


def test_a_setting_above_zero_is_the_one_candidate() -> None:
    assert candidates(3, 20) == (3,)


def test_a_setting_at_zero_searches_the_halvings_of_the_top() -> None:
    assert candidates(0, 20) == (20, 10, 5, 2, 1)


# --- rerank --------------------------------------------------------------------------


async def test_rerank_at_zero_searches_batch_sizes_then_threads(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reranker = await open_reranker(four_thread_settings(settings_factory), caplog)

    line = calibrated_lines(caplog)["rerank"]
    trials = tried(line)
    assert trials[:5] == [(4, 20), (4, 10), (4, 5), (4, 2), (4, 1)]
    assert [threads for threads, _ in trials[5:]] == [2, 1]
    assert len({batch for _, batch in trials[5:]}) == 1  # the best so far, once
    assert line["cut_short"] is False
    assert fake_weights.loaded == [f"reranker:{RERANKER}"] * 3  # one per thread count
    chosen = line["chosen"]
    assert reranker.scope.threads == chosen["threads"]
    assert reranker.with_batch_size(1).scope is reranker.scope
    assert fake_weights.batch_sizes[-1] == ("predict", trials[-1][1])


async def test_rerank_settings_above_zero_pin_both_and_time_nothing(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = four_thread_settings(
        settings_factory,
        components__reranker__local__threads=3,
        components__reranker__local__batch_size=7,
    )
    reranker = await open_reranker(settings, caplog)

    assert calibrated_lines(caplog) == {}
    assert fake_weights.batch_sizes == []  # no request was served
    assert reranker.scope.threads == 3
    assert len(fake_weights.loaded) == 1


async def test_a_pinned_rerank_batch_size_leaves_the_threads_searched(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = four_thread_settings(
        settings_factory, components__reranker__local__batch_size=7
    )
    await open_reranker(settings, caplog)

    assert tried(calibrated_lines(caplog)["rerank"]) == [(4, 7), (2, 7), (1, 7)]


async def test_the_calibration_limits_come_from_the_settings(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = four_thread_settings(
        settings_factory, calibration_budget_s=1e-9, calibration_rounds=3
    )
    reranker = await open_reranker(settings, caplog)

    line = calibrated_lines(caplog)["rerank"]
    assert tried(line) == [(4, 20)]  # the first trial always runs
    assert (line["budget_s"], line["cut_short"]) == (1e-9, True)
    assert fake_weights.batch_sizes == [("predict", 20)] * 4  # warm-up, 3 rounds
    assert reranker.scope.threads == 4


async def test_the_rerankers_calibration_passages_bound_the_batch_sizes_tried(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = four_thread_settings(
        settings_factory, components__reranker__local__calibration_passages=8
    )
    await open_reranker(settings, caplog)

    trials = tried(calibrated_lines(caplog)["rerank"])
    assert trials[:4] == [(4, 8), (4, 4), (4, 2), (4, 1)]
    assert ("predict", 8) in fake_weights.batch_sizes  # a request of 8 passages


async def test_rerank_calibration_reads_no_stages_setting(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph = rag_graph(retrieve_top_k=7, rerank_top_k=3)
    await open_reranker(four_thread_settings(settings_factory, graph=graph), caplog)

    trials = tried(calibrated_lines(caplog)["rerank"])
    assert trials[0] == (4, 20)  # the reranker's 20 passages, not retrieve's 7


async def test_the_rerank_sample_is_one_question_and_its_calibration_passages(
    pool: ThreadPoolExecutor,
) -> None:
    model = RecordingCrossEncoder()
    reranker = CrossEncoderReranker(
        model, pool, model_name="m", batch_size=4, scope=NativeThreadScope(1)
    )
    stage = CalibratedRerankStage(TORCH_MODEL, pool, passages=12)

    await stage.serve_sample(reranker)

    sample = calibration_sample(12)
    [pairs] = model.pairs
    assert pairs == [(sample.question, text) for text in sample.passages]


# --- embed ---------------------------------------------------------------------------


async def test_embed_at_zero_searches_threads_at_the_configured_batch_size(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = four_thread_settings(
        settings_factory, components__embedder__local__batch_size=16
    )
    embedder = await open_embedder(settings, caplog)

    line = calibrated_lines(caplog)["embed"]
    assert tried(line) == [(4, 16), (2, 16), (1, 16)]
    assert embedder.scope.threads == line["chosen"]["threads"]


async def test_embed_threads_above_zero_time_nothing(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = four_thread_settings(
        settings_factory, components__embedder__local__threads=2
    )
    embedder = await open_embedder(settings, caplog)

    assert calibrated_lines(caplog) == {}
    assert embedder.scope.threads == 2
    assert fake_weights.batch_sizes == []


@pytest.mark.parametrize("prefix", ["query: ", ""])
async def test_the_embed_sample_is_one_question_embedded_as_a_query(
    fake_weights: FakeSentenceTransformersModule,
    pool: ThreadPoolExecutor,
    prefix: str,
) -> None:
    stage = CalibratedEmbedStage(TORCH_MODEL, pool, query_prefix=prefix)

    # The adapter loaded for a trial carries the embedder's prefix.
    await stage.serve_sample(await stage.loaded(threads=1))

    assert fake_weights.encoded == [[prefix + calibration_sample(1).question]]


# --- the settings that pin a tuning --------------------------------------------------


def test_the_rerank_stage_is_pinned_by_its_threads_and_batch_size(
    pool: ThreadPoolExecutor,
) -> None:
    stage = CalibratedRerankStage(TORCH_MODEL, pool, passages=20)

    pinned = stage.pinned_settings(StageTuning(threads=2, batch_size=5))

    assert pinned == {"threads": 2, "batch_size": 5}  # in its local block


def test_the_embed_stage_is_pinned_by_its_threads_only(
    pool: ThreadPoolExecutor,
) -> None:
    stage = CalibratedEmbedStage(TORCH_MODEL, pool, query_prefix="query: ")

    pinned = stage.pinned_settings(StageTuning(threads=3, batch_size=64))

    assert pinned == {"threads": 3}


def test_pinned_settings_read_back_as_the_tuning(
    settings_factory: SettingsFactory,
    pool: ThreadPoolExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = CalibratedRerankStage(TORCH_MODEL, pool, passages=20)
    pinned = stage.pinned_settings(StageTuning(threads=2, batch_size=5))
    for setting, value in pinned.items():
        path = ("components", "reranker", "local", setting)
        monkeypatch.setenv(RagSettings.variable_of(*path), str(value))

    local = settings_factory().components["reranker"].local

    assert (local.threads, local.batch_size) == (2, 5)


# --- recording models ----------------------------------------------------------------


class RecordingCrossEncoder:
    def __init__(self) -> None:
        self.pairs: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> list[float]:
        self.pairs.append(list(pairs))
        return [0.5 for _ in pairs]
