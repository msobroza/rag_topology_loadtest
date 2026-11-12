"""The in-process sentence-transformers adapters over injected doubles: no weights.

Every adapter takes its heavy collaborator through the constructor, so each test
hands in a small fake.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from rag_load_test.adapters.local_models import sentence_transformer_models
from rag_load_test.adapters.local_models.local_model import LocalModel
from rag_load_test.adapters.local_models.native_threads import NativeThreadScope
from rag_load_test.adapters.local_models.sentence_transformer_models import (
    CrossEncoderReranker,
    SentenceTransformerEmbedder,
)
from rag_load_test.domain.passages import ScoredPassage


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = sentence_transformer_models.model_executor(workers=2)
    yield pool
    pool.shutdown(wait=True)


def test_model_executor_runs_the_given_number_of_workers() -> None:
    pool = sentence_transformer_models.model_executor(workers=3)
    try:
        assert pool._max_workers == 3
    finally:
        pool.shutdown(wait=True)


def test_model_executor_runs_on_thread_start_in_each_new_worker() -> None:
    started: list[str] = []

    def record() -> None:
        started.append(threading.current_thread().name)

    pool = sentence_transformer_models.model_executor(workers=1, on_thread_start=record)
    try:
        worker = pool.submit(lambda: threading.current_thread().name).result()
    finally:
        pool.shutdown(wait=True)
    assert started == [worker]
    assert worker.startswith("rag-model")


class RecordingScope(NativeThreadScope):
    """A real scope that also records the thread each call ran on."""

    def __init__(self, threads: int) -> None:
        super().__init__(threads)
        self.ran_on: list[str] = []

    def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        self.ran_on.append(threading.current_thread().name)
        return super().run(fn, *args)


def scored_passages(*ids: str) -> list[ScoredPassage]:
    return [ScoredPassage(id=i, text=f"text of {i}", retrieval_score=0.5) for i in ids]


# --- SentenceTransformerEmbedder -------------------------------------------


class FakeSentenceTransformer:
    """``encode`` double: vector = [len(text), 1.0]; records every call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(
        self, texts: list[str], *, batch_size: int, normalize_embeddings: bool = False
    ) -> np.ndarray:
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)


@pytest.fixture
def st_model() -> FakeSentenceTransformer:
    return FakeSentenceTransformer()


@pytest.fixture
def st_scope() -> RecordingScope:
    return RecordingScope(2)


@pytest.fixture
def st_embedder(
    st_model: FakeSentenceTransformer,
    executor: ThreadPoolExecutor,
    st_scope: RecordingScope,
) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        st_model, executor, model_name="fake-st", batch_size=8, scope=st_scope
    )


async def test_st_embedder_encodes_inside_its_scope_on_a_worker(
    st_embedder: SentenceTransformerEmbedder, st_scope: RecordingScope
) -> None:
    await st_embedder.embed(["a"])

    assert st_embedder.scope is st_scope
    [thread] = st_scope.ran_on
    assert thread.startswith("rag-model")


async def test_st_embedder_returns_plain_lists_and_normalizes(
    st_embedder: SentenceTransformerEmbedder, st_model: FakeSentenceTransformer
) -> None:
    vectors = await st_embedder.embed(("ab", "abcd"))
    assert vectors == [[2.0, 1.0], [4.0, 1.0]]
    assert type(vectors) is list
    assert all(type(v) is list for v in vectors)
    assert all(type(x) is float for v in vectors for x in v)
    assert st_model.calls == [
        {"texts": ["ab", "abcd"], "batch_size": 8, "normalize_embeddings": True}
    ]


async def test_st_embedder_empty_input_skips_model_and_ready_names_model(
    st_embedder: SentenceTransformerEmbedder, st_model: FakeSentenceTransformer
) -> None:
    assert await st_embedder.embed([]) == []
    assert st_model.calls == []
    assert await st_embedder.ready() == (True, "fake-st")


# --- CrossEncoderReranker --------------------------------------------------


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


# --- from_pretrained: what reaches the library ------------------------------


class FakeLoadingModule(ModuleType):
    """Stands in for ``sentence_transformers``; records each model's load arguments."""

    def __init__(self) -> None:
        super().__init__("sentence_transformers")
        self.loads: list[tuple[str, str, dict[str, Any]]] = []
        loads = self.loads

        class SentenceTransformer:
            def __init__(self, source: str, **kwargs: Any) -> None:
                loads.append(("SentenceTransformer", source, kwargs))

        class CrossEncoder:
            def __init__(self, source: str, **kwargs: Any) -> None:
                loads.append(("CrossEncoder", source, kwargs))

        self.SentenceTransformer = SentenceTransformer
        self.CrossEncoder = CrossEncoder


@pytest.fixture
def fake_loading(monkeypatch: pytest.MonkeyPatch) -> FakeLoadingModule:
    module = FakeLoadingModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


OPENVINO_EXPORT = LocalModel("org/name", "models/openvino/name", "openvino")
OPENVINO_ARGUMENTS = {
    "device": "cpu",
    "backend": "openvino",
    "model_kwargs": {"ov_config": {"INFERENCE_NUM_THREADS": "3"}},
}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (LocalModel("org/name", "org/name", "torch"), {"device": "cpu"}),
        (OPENVINO_EXPORT, OPENVINO_ARGUMENTS),
    ],
)
async def test_from_pretrained_loads_the_source_with_the_runtime_arguments(
    fake_loading: FakeLoadingModule,
    executor: ThreadPoolExecutor,
    model: LocalModel,
    expected: dict[str, Any],
) -> None:
    embedder = SentenceTransformerEmbedder.from_pretrained(
        model, executor, batch_size=8, threads=3
    )
    reranker = CrossEncoderReranker.from_pretrained(
        model, executor, batch_size=4, threads=3
    )

    assert fake_loading.loads == [
        ("SentenceTransformer", model.source, expected),
        ("CrossEncoder", model.source, expected),
    ]
    assert await embedder.ready() == (True, "org/name")
    assert await reranker.ready() == (True, "org/name")
    assert embedder.scope.threads == reranker.scope.threads == 3
