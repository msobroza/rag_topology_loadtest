"""``ElasticsearchVectorStore`` over a fake LangChain store: no Elasticsearch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rag_load_test.adapters.vector_store.elasticsearch_store import (
    ElasticsearchVectorStore,
)
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.passages import Passage

INDEX = "passages_under_test"  # any name: the adapter must use the one it is given


# --- ElasticsearchVectorStore ----------------------------------------------


def es_hit(passage_id: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"_id": passage_id, "_source": {"text": text, "metadata": metadata}}


class FakeLangChainStore:
    """Double for LangChain's ``AsyncElasticsearchStore``: scripted hits or a failure.

    ``client`` exposes only the elasticsearch-py coroutines the adapter uses.
    """

    def __init__(
        self,
        hits: Sequence[tuple[dict[str, Any], float]] = (),
        *,
        exists: bool = True,
        count: int = 0,
        lexical_hits: Sequence[dict[str, Any]] = (),
        failure: Exception | None = None,
    ) -> None:
        self.hits, self.exists, self.count = list(hits), exists, count
        self.lexical_hits, self.failure = list(lexical_hits), failure
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.count_calls: list[str] = []
        self.lexical_calls: list[dict[str, Any]] = []
        self.closed = False
        indices = SimpleNamespace(exists=self._indices_exist)
        self.client = SimpleNamespace(
            indices=indices, count=self._count_docs, search=self._search_text
        )

    async def _indices_exist(self, *, index: str) -> bool:
        self._maybe_fail()
        return self.exists

    async def _count_docs(self, *, index: str) -> dict[str, int]:
        self.count_calls.append(index)
        return {"count": self.count}

    async def _search_text(
        self, *, index: str, query: dict[str, Any], size: int
    ) -> dict[str, Any]:
        self._maybe_fail()
        self.lexical_calls.append({"index": index, "query": query, "size": size})
        return {"hits": {"hits": self.lexical_hits[:size]}}

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

    async def aclose(self) -> None:
        self.closed = True

    def _maybe_fail(self) -> None:
        if self.failure is not None:
            raise self.failure


async def test_es_upsert_passes_text_vector_pairs_metadatas_and_ids() -> None:
    fake = FakeLangChainStore()
    passages = [
        Passage(id="p1", text="alpha", metadata={"topic": "hr"}),
        Passage(id="p2", text="beta"),
    ]
    store = ElasticsearchVectorStore(fake, INDEX)
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
    hits = await ElasticsearchVectorStore(fake, INDEX).query((0.5, 0.5), top_k=2)
    assert [(h.id, h.text, h.metadata) for h in hits] == [
        ("p9", "nine", {"topic": "it"}),
        ("p4", "four", {}),
    ]
    scores = [h.retrieval_score for h in hits]
    assert scores == [0.75, 0.25]
    assert scores == sorted(scores, reverse=True)
    assert all(type(h.retrieval_score) is float for h in hits)
    assert all(h.rerank_score is None for h in hits)
    assert fake.search_calls == [{"embedding": [0.5, 0.5], "k": 2}]


async def test_es_lexical_query_is_a_bm25_match_on_the_text_field() -> None:
    fake = FakeLangChainStore(
        lexical_hits=[
            {**es_hit("p7", "vacation days", {"topic": "hr"}), "_score": 3.5},
            {**es_hit("p2", "vacation", {}), "_score": 1.25},
        ]
    )
    hits = await ElasticsearchVectorStore(fake, INDEX).lexical_query("vacation", 2)
    assert fake.lexical_calls == [
        {"index": INDEX, "query": {"match": {"text": "vacation"}}, "size": 2}
    ]
    assert [(h.id, h.text, h.metadata, h.retrieval_score) for h in hits] == [
        ("p7", "vacation days", {"topic": "hr"}, 3.5),
        ("p2", "vacation", {}, 1.25),
    ]
    assert all(type(h.retrieval_score) is float for h in hits)
    assert all(h.rerank_score is None for h in hits)


async def test_es_lexical_query_without_a_match_is_empty() -> None:
    fake = FakeLangChainStore()
    assert await ElasticsearchVectorStore(fake, INDEX).lexical_query("x", 5) == []


@pytest.mark.parametrize(("exists", "expected"), [(False, 0), (True, 17)])
async def test_es_count_is_zero_without_index(exists: bool, expected: int) -> None:
    fake = FakeLangChainStore(exists=exists, count=17)
    assert await ElasticsearchVectorStore(fake, INDEX).count() == expected
    assert fake.count_calls == ([INDEX] if exists else [])


@pytest.mark.parametrize(
    ("exists", "expected"),
    [
        (False, (False, f"0 passages in {INDEX}")),
        (True, (True, f"17 passages in {INDEX}")),
    ],
)
async def test_es_ready_reports_the_passage_count_and_index(
    exists: bool, expected: tuple[bool, str]
) -> None:
    fake = FakeLangChainStore(exists=exists, count=17)
    assert await ElasticsearchVectorStore(fake, INDEX).ready() == expected


async def test_es_aclose_closes_the_underlying_store() -> None:
    fake = FakeLangChainStore()
    await ElasticsearchVectorStore(fake, INDEX).aclose()
    assert fake.closed


StoreOperation = Callable[[ElasticsearchVectorStore], Awaitable[Any]]


@pytest.mark.parametrize(
    "operation",
    [
        lambda store: store.upsert([Passage(id="p1", text="alpha")], [[0.1]]),
        lambda store: store.query([0.1], top_k=1),
        lambda store: store.lexical_query("alpha", top_k=1),
        lambda store: store.count(),
        lambda store: store.ready(),
    ],
    ids=["upsert", "query", "lexical_query", "count", "ready"],
)
async def test_es_failures_surface_as_vector_store_errors(
    operation: StoreOperation,
) -> None:
    boom = ConnectionError("es refused")
    store = ElasticsearchVectorStore(FakeLangChainStore(failure=boom), INDEX)
    with pytest.raises(RagDependencyError) as info:
        await operation(store)
    assert info.value.component == "vector_store"
    assert info.value.target == INDEX
    assert info.value.kind == "unavailable"
    assert info.value.detail == "ConnectionError: es refused"
    assert info.value.__cause__ is boom
