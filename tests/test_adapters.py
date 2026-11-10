"""Unit tests for ``rag_load_test.adapters`` with injected doubles.

No model weights, no Elasticsearch, no OpenAI: every adapter takes its heavy
collaborator through the constructor, so each test hands in a small fake.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Awaitable, Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import openai
import pytest

from rag_load_test import adapters
from rag_load_test.adapters import (
    INDEX,
    CrossEncoderReranker,
    ElasticsearchVectorStore,
    NoReranker,
    OpenAIChatModel,
    SentenceTransformerEmbedder,
)
from rag_load_test.models import Passage, RagDependencyError, ScoredPassage

HEAVY_MODULES = ("sentence_transformers", "torch", "elasticsearch", "openai")


@pytest.fixture
def executor() -> Iterator[ThreadPoolExecutor]:
    pool = adapters.model_executor()
    yield pool
    pool.shutdown(wait=True)


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
def st_embedder(
    st_model: FakeSentenceTransformer, executor: ThreadPoolExecutor
) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        st_model, executor, model_name="fake-st", batch_size=8
    )


async def test_st_embedder_returns_plain_lists_and_normalizes(
    st_embedder: SentenceTransformerEmbedder, st_model: FakeSentenceTransformer
) -> None:
    vectors = await st_embedder.embed(("ab", "abcd"))
    assert vectors == [[2.0, 1.0], [4.0, 1.0]]
    assert type(vectors) is list and all(type(v) is list for v in vectors)
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


# --- CrossEncoderReranker / NoReranker --------------------------------------


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
    reranker = CrossEncoderReranker(model, executor, model_name="fake-ce", batch_size=4)
    ranked = await reranker.rerank("q", scored_passages("a", "b", "c"), top_n=2)
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
    reranker = CrossEncoderReranker(model, executor, model_name="fake-ce")
    assert await reranker.rerank("q", [], top_n=3) == []
    assert model.calls == []
    assert await reranker.ready() == (True, "fake-ce")


async def test_no_reranker_keeps_order_and_truncates() -> None:
    ranked = await NoReranker().rerank("q", scored_passages("a", "b", "c"), top_n=2)
    assert [p.id for p in ranked] == ["a", "b"]
    assert all(p.rerank_score is None for p in ranked)
    assert await NoReranker().ready() == (True, "none")


# --- ElasticsearchVectorStore ----------------------------------------------


def es_hit(passage_id: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"_id": passage_id, "_source": {"text": text, "metadata": metadata}}


class FakeLangChainStore:
    """Double for LangChain's ``AsyncElasticsearchStore``: scripted hits or a failure.

    ``client`` exposes only the two elasticsearch-py coroutines the adapter uses.
    """

    def __init__(
        self,
        hits: Sequence[tuple[dict[str, Any], float]] = (),
        *,
        exists: bool = True,
        count: int = 0,
        failure: Exception | None = None,
    ) -> None:
        self.hits, self.exists, self.count = list(hits), exists, count
        self.failure = failure
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.count_calls: list[str] = []
        indices = SimpleNamespace(exists=self._indices_exist)
        self.client = SimpleNamespace(indices=indices, count=self._count_docs)

    async def _indices_exist(self, *, index: str) -> bool:
        self._maybe_fail()
        return self.exists

    async def _count_docs(self, *, index: str) -> dict[str, int]:
        self.count_calls.append(index)
        return {"count": self.count}

    async def aadd_embeddings(
        self,
        pairs: Sequence[tuple[str, list[float]]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> list[str]:
        self._maybe_fail()
        self.add_calls.append(
            {"pairs": list(pairs), "metadatas": metadatas, "ids": ids}
        )
        return list(ids)

    async def asimilarity_search_by_vector_with_relevance_scores(
        self, embedding: list[float], k: int, doc_builder: Callable[..., Any]
    ) -> list[tuple[Any, float]]:
        self._maybe_fail()
        self.search_calls.append({"embedding": embedding, "k": k})
        return [(doc_builder(hit), score) for hit, score in self.hits[:k]]

    def _maybe_fail(self) -> None:
        if self.failure is not None:
            raise self.failure


async def test_es_upsert_passes_text_vector_pairs_metadatas_and_ids() -> None:
    fake = FakeLangChainStore()
    passages = [
        Passage(id="p1", text="alpha", metadata={"topic": "hr"}),
        Passage(id="p2", text="beta"),
    ]
    store = ElasticsearchVectorStore(fake)
    await store.upsert(passages, [(0.1, 0.2), [0.3, 0.4]])
    await store.upsert([], [])  # nothing to write: no call at all
    assert fake.add_calls == [
        {
            "pairs": [("alpha", [0.1, 0.2]), ("beta", [0.3, 0.4])],
            "metadatas": [{"topic": "hr"}, {}],
            "ids": ["p1", "p2"],
        }
    ]


async def test_es_query_builds_scored_passages_from_hits() -> None:
    fake = FakeLangChainStore(
        [
            (es_hit("p9", "nine", {"topic": "it"}), np.float32(0.75)),
            (es_hit("p4", "four", {}), 0.25),
        ]
    )
    hits = await ElasticsearchVectorStore(fake).query((0.5, 0.5), top_k=2)
    assert [(h.id, h.text, h.metadata) for h in hits] == [
        ("p9", "nine", {"topic": "it"}),
        ("p4", "four", {}),
    ]
    scores = [h.retrieval_score for h in hits]
    assert scores == [0.75, 0.25] and scores == sorted(scores, reverse=True)
    assert all(type(h.retrieval_score) is float for h in hits)
    assert all(h.rerank_score is None for h in hits)
    assert fake.search_calls == [{"embedding": [0.5, 0.5], "k": 2}]


@pytest.mark.parametrize(("exists", "expected"), [(False, 0), (True, 17)])
async def test_es_count_is_zero_without_index(exists: bool, expected: int) -> None:
    fake = FakeLangChainStore(exists=exists, count=17)
    assert await ElasticsearchVectorStore(fake).count() == expected
    assert fake.count_calls == ([INDEX] if exists else [])


StoreOperation = Callable[[ElasticsearchVectorStore], Awaitable[Any]]


@pytest.mark.parametrize(
    "operation",
    [
        lambda store: store.upsert([Passage(id="p1", text="alpha")], [[0.1]]),
        lambda store: store.query([0.1], top_k=1),
        lambda store: store.count(),
    ],
    ids=["upsert", "query", "count"],
)
async def test_es_failures_surface_as_vector_store_errors(
    operation: StoreOperation,
) -> None:
    boom = ConnectionError("es refused")
    store = ElasticsearchVectorStore(FakeLangChainStore(failure=boom))
    with pytest.raises(RagDependencyError) as info:
        await operation(store)
    assert info.value.component == "vector_store"
    assert info.value.target == INDEX
    assert info.value.kind == "unavailable"
    assert info.value.detail == "ConnectionError: es refused"
    assert info.value.__cause__ is boom


# --- OpenAIChatModel -------------------------------------------------------


class FakeChatCompletions:
    """``chat.completions.create`` double: returns ``outcome`` or raises it."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def fake_openai_client(outcome: Any) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=FakeChatCompletions(outcome))
    )


def completion(content: str | None) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def test_openai_chat_returns_content_and_passes_model_at_temperature_zero() -> (
    None
):
    client = fake_openai_client(completion("Paris."))
    messages = [{"role": "user", "content": "Capital of France?"}]
    assert await OpenAIChatModel(client, "gpt-test").generate(messages) == "Paris."
    assert client.chat.completions.calls == [
        {"model": "gpt-test", "messages": messages, "temperature": 0}
    ]


def _openai_request() -> httpx.Request:
    return httpx.Request("POST", "http://x")


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (openai.APITimeoutError(request=_openai_request()), "timeout"),
        (
            openai.APIStatusError(
                "boom",
                response=httpx.Response(500, request=_openai_request()),
                body=None,
            ),
            "unavailable",
        ),
    ],
    ids=["timeout", "status"],
)
async def test_openai_errors_map_to_llm_dependency_errors(
    error: Exception, kind: str
) -> None:
    client = fake_openai_client(error)
    with pytest.raises(RagDependencyError) as info:
        await OpenAIChatModel(client, "gpt-test").generate([])
    assert info.value.component == "llm"
    assert info.value.kind == kind
    assert info.value.target == "gpt-test"
    assert info.value.__cause__ is error


# --- import cost -----------------------------------------------------------


def test_importing_adapters_leaves_heavy_libraries_unimported() -> None:
    # Fresh interpreter: this process may already have torch/openai loaded.
    code = (
        "import sys; import rag_load_test.adapters; "
        f"print(sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules))"
    )
    command = [sys.executable, "-c", code]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
