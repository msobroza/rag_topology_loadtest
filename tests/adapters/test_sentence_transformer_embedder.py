"""The ``local`` embedder over injected doubles: no weights.

It takes its heavy collaborator through the constructor, so each test hands in
a small fake.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from local_model_fakes import (
    RUNTIME_LOADS,
    FakeLoadingModule,
    RecordingScope,
)
from rag_load_test.adapters.embedder.local.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from rag_load_test.adapters.local_models.local_model import LocalModel

pytest_plugins = ["local_model_fixtures"]


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


@pytest.mark.parametrize(("model", "expected"), RUNTIME_LOADS)
async def test_from_pretrained_loads_the_source_with_the_runtime_arguments(
    fake_loading: FakeLoadingModule,
    executor: ThreadPoolExecutor,
    model: LocalModel,
    expected: dict[str, Any],
) -> None:
    embedder = SentenceTransformerEmbedder.from_pretrained(
        model, executor, batch_size=8, threads=3
    )

    assert fake_loading.loads == [("SentenceTransformer", model.source, expected)]
    assert await embedder.ready() == (True, "org/name")
    assert embedder.scope.threads == 3
