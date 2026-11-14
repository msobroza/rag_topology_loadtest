"""The ``local`` reranker over injected doubles: no weights."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from local_model_fakes import (
    RUNTIME_LOADS,
    FakeLoadingModule,
    RecordingScope,
    scored_passages,
)
from rag_load_test.adapters.reranker.local.cross_encoder_reranker import (
    CrossEncoderReranker,
)
from rag_load_test.infrastructure.local_models.local_model import LocalModel

pytest_plugins = ["local_model_fixtures"]


class FakeCrossEncoder:
    """``predict`` double returning scripted scores as a float32 array."""

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.calls: list[dict[str, Any]] = []

    def predict(self, pairs: list[tuple[str, str]], *, batch_size: int) -> np.ndarray:
        self.calls.append({"pairs": list(pairs), "batch_size": batch_size})
        return np.array(self.scores[: len(pairs)], dtype=np.float32)


async def test_cross_encoder_orders_by_score_and_truncates(
    executor: ThreadPoolExecutor,
) -> None:
    model = FakeCrossEncoder([0.25, 0.75, 0.5])  # exact in float32
    scope = RecordingScope(3)
    reranker = CrossEncoderReranker(
        model, executor, model_name="fake-ce", batch_size=4, scope=scope
    )
    ranked = await reranker.rerank("q", scored_passages("a", "b", "c"), top_n=2)
    assert [t.startswith("rag-model") for t in scope.ran_on] == [True]
    assert [p.id for p in ranked] == ["b", "c"]
    assert [p.rerank_score for p in ranked] == [0.75, 0.5]
    assert all(type(p.rerank_score) is float for p in ranked)
    assert all(p.retrieval_score == 0.5 for p in ranked)
    expected_pairs = [("q", "text of a"), ("q", "text of b"), ("q", "text of c")]
    assert model.calls == [{"pairs": expected_pairs, "batch_size": 4}]


async def test_cross_encoder_empty_input_skips_model(
    executor: ThreadPoolExecutor,
) -> None:
    model = FakeCrossEncoder([])
    reranker = CrossEncoderReranker(
        model, executor, model_name="fake-ce", batch_size=32, scope=RecordingScope(1)
    )
    assert await reranker.rerank("q", [], top_n=3) == []
    assert model.calls == []
    assert await reranker.ready() == (True, "fake-ce")


@pytest.mark.parametrize(("model", "expected"), RUNTIME_LOADS)
async def test_from_pretrained_loads_the_source_with_the_runtime_arguments(
    fake_loading: FakeLoadingModule,
    executor: ThreadPoolExecutor,
    model: LocalModel,
    expected: dict[str, Any],
) -> None:
    reranker = CrossEncoderReranker.from_pretrained(
        model, executor, batch_size=4, threads=3
    )

    assert fake_loading.loads == [("CrossEncoder", model.source, expected)]
    assert await reranker.ready() == (True, "org/name")
    assert reranker.scope.threads == 3
