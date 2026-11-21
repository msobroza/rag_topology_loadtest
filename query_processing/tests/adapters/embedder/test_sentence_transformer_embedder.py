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
from query_processing.adapters.embedder.local.sentence_transformer_embedder import (
    EncodeOptions,
    SentenceTransformerEmbedder,
)
from query_processing.infrastructure.local_models.local_model import LocalModel

pytest_plugins = ["local_model_fixtures"]


class FakeSentenceTransformer:
    """``encode`` double: vector = [len(text), 1.0]; records every call.

    ``show_progress_bar`` has no default, so an ``encode`` call without it fails.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
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
        st_model,
        executor,
        model_name="fake-st",
        scope=st_scope,
        query_prefix="query: ",  # a text to store must never get it
        encode_options=EncodeOptions(batch_size=8, normalize_embeddings=False),
    )


async def test_st_embedder_encodes_inside_its_scope_on_a_worker(
    st_embedder: SentenceTransformerEmbedder, st_scope: RecordingScope
) -> None:
    await st_embedder.embed_documents(["a"])

    assert st_embedder.scope is st_scope
    [thread] = st_scope.ran_on
    assert thread.startswith("rag-model")


async def test_st_embedder_returns_plain_lists_not_normalised_by_default(
    st_embedder: SentenceTransformerEmbedder, st_model: FakeSentenceTransformer
) -> None:
    vectors = await st_embedder.embed_documents(("ab", "abcd"))
    assert vectors == [[2.0, 1.0], [4.0, 1.0]]
    assert type(vectors) is list
    assert all(type(v) is list for v in vectors)
    assert all(type(x) is float for v in vectors for x in v)
    assert st_model.calls == [
        {
            "texts": ["ab", "abcd"],
            "batch_size": 8,
            "normalize_embeddings": False,
            # No tqdm bar on stderr per call (the Apps log at INFO).
            "show_progress_bar": False,
        }
    ]


async def test_st_embedder_empty_input_skips_model_and_ready_names_model(
    st_embedder: SentenceTransformerEmbedder, st_model: FakeSentenceTransformer
) -> None:
    assert await st_embedder.embed_documents([]) == []
    assert st_model.calls == []
    assert await st_embedder.ready() == (True, "fake-st")


async def test_a_query_is_encoded_alone_with_the_prefix_in_front(
    st_embedder: SentenceTransformerEmbedder,
    st_model: FakeSentenceTransformer,
    st_scope: RecordingScope,
) -> None:
    vector = await st_embedder.embed_query("ab")

    assert vector == [9.0, 1.0]  # len("query: ab")
    # As its encode options say: not normalised here, and never a progress bar.
    assert st_model.calls == [
        {
            "texts": ["query: ab"],
            "batch_size": 8,
            "normalize_embeddings": False,
            "show_progress_bar": False,
        }
    ]
    [thread] = st_scope.ran_on
    assert thread.startswith("rag-model")


@pytest.mark.parametrize(("model", "expected"), RUNTIME_LOADS)
async def test_from_pretrained_loads_the_source_with_the_runtime_arguments(
    fake_loading: FakeLoadingModule,
    executor: ThreadPoolExecutor,
    model: LocalModel,
    expected: dict[str, Any],
) -> None:
    embedder = SentenceTransformerEmbedder.from_pretrained(
        model, executor, batch_size=8, threads=3, query_prefix="query: "
    )

    assert fake_loading.loads == [("SentenceTransformer", model.source, expected)]
    assert await embedder.ready() == (True, "org/name")
    assert embedder.scope.threads == 3


async def test_with_batch_size_shares_the_model_and_changes_only_the_batch(
    st_embedder: SentenceTransformerEmbedder,
    st_model: FakeSentenceTransformer,
    st_scope: RecordingScope,
) -> None:
    rebatched = st_embedder.with_batch_size(3)
    await rebatched.embed_documents(["a"])
    await st_embedder.embed_documents(["b"])
    await rebatched.embed_query("c")

    assert type(rebatched) is SentenceTransformerEmbedder
    assert [call["batch_size"] for call in st_model.calls] == [3, 8, 3]
    assert st_model.calls[-1]["texts"] == ["query: c"]  # the prefix is kept
    assert rebatched.scope is st_scope
    assert await rebatched.ready() == (True, "fake-st")
    assert all(thread.startswith("rag-model") for thread in st_scope.ran_on)


async def test_st_embedder_passes_normalize_embeddings_to_encode_and_keeps_it(
    st_model: FakeSentenceTransformer,
    executor: ThreadPoolExecutor,
    st_scope: RecordingScope,
) -> None:
    embedder = SentenceTransformerEmbedder(
        st_model,
        executor,
        model_name="fake-st",
        scope=st_scope,
        query_prefix="",
        encode_options=EncodeOptions(batch_size=8, normalize_embeddings=True),
    )

    await embedder.embed_documents(["a"])
    await embedder.with_batch_size(3).embed_documents(["b"])

    assert [call["normalize_embeddings"] for call in st_model.calls] == [True, True]
