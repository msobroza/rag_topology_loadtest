"""The rerank and embed stages, and how their factories calibrate them from the settings.

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

import numpy as np
import pytest

from rag_load_test.adapters.embedder.local.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from rag_load_test.adapters.reranker.local.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from rag_load_test.composition.calibrated_stages import (
    EmbedStage,
    RerankStage,
    candidates,
)
from rag_load_test.composition.composition_root import CompositionRoot
from rag_load_test.config.rag_settings import RagSettings
from rag_load_test.domain.calibration import calibration_sample
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
        generation_backend="fake",
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
            reranker = await composition.local_reranker()
    assert isinstance(reranker, CrossEncoderReranker)
    return reranker


async def open_embedder(
    settings: RagSettings, caplog: pytest.LogCaptureFixture
) -> SentenceTransformerEmbedder:
    with caplog.at_level("INFO", logger="rag_load_test"):
        async with CompositionRoot(settings) as composition:
            embedder = await composition.local_embedder()
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
        settings_factory, rerank_threads=3, rerank_batch_size=7
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
    settings = four_thread_settings(settings_factory, rerank_batch_size=7)
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


async def test_the_rerank_sample_is_one_question_and_top_k_retrieve_passages(
    settings_factory: SettingsFactory, pool: ThreadPoolExecutor
) -> None:
    model = RecordingCrossEncoder()
    reranker = CrossEncoderReranker(
        model, pool, model_name="m", batch_size=4, scope=NativeThreadScope(1)
    )
    settings = settings_factory(top_k_retrieve=12, top_k_rerank=3)

    await RerankStage(TORCH_MODEL, pool, settings).serve_sample(reranker)

    sample = calibration_sample(12)
    [pairs] = model.pairs
    assert pairs == [(sample.question, text) for text in sample.passages]


# --- embed ---------------------------------------------------------------------------


async def test_embed_at_zero_searches_threads_at_the_configured_batch_size(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = four_thread_settings(settings_factory, embedding_batch_size=16)
    embedder = await open_embedder(settings, caplog)

    line = calibrated_lines(caplog)["embed"]
    assert tried(line) == [(4, 16), (2, 16), (1, 16)]
    assert embedder.scope.threads == line["chosen"]["threads"]


async def test_embed_threads_above_zero_time_nothing(
    settings_factory: SettingsFactory,
    fake_weights: FakeSentenceTransformersModule,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = four_thread_settings(settings_factory, embedding_threads=2)
    embedder = await open_embedder(settings, caplog)

    assert calibrated_lines(caplog) == {}
    assert embedder.scope.threads == 2
    assert fake_weights.batch_sizes == []


async def test_the_embed_sample_is_one_question_with_the_query_prefix(
    settings_factory: SettingsFactory, pool: ThreadPoolExecutor
) -> None:
    model = RecordingSentenceTransformer()
    embedder = SentenceTransformerEmbedder(
        model, pool, model_name="m", batch_size=4, scope=NativeThreadScope(1)
    )
    settings = settings_factory(embedding_query_prefix="query: ")

    await EmbedStage(TORCH_MODEL, pool, settings).serve_sample(embedder)

    assert model.texts == [[f"query: {calibration_sample(1).question}"]]


# --- recording models ----------------------------------------------------------------


class RecordingCrossEncoder:
    def __init__(self) -> None:
        self.pairs: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> list[float]:
        self.pairs.append(list(pairs))
        return [0.5 for _ in pairs]


class RecordingSentenceTransformer:
    def __init__(self) -> None:
        self.texts: list[list[str]] = []

    def encode(
        self, texts: list[str], *, batch_size: int, normalize_embeddings: bool
    ) -> np.ndarray:
        self.texts.append(list(texts))
        return np.array([[1.0, 0.0] for _ in texts])
