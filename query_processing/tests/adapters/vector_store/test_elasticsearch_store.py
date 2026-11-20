"""``ElasticsearchVectorStore`` over a fake LangChain store: no Elasticsearch."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from elastic_transport import JsonSerializer
from elasticsearch.helpers.vectorstore import AsyncDenseVectorStrategy, DistanceMetric
from langchain_elasticsearch import AsyncElasticsearchStore
from query_processing_core.retrieval.metadata_filter import MetadataCondition
from query_processing_core.retrieval.vector_store_record import VectorStoreRecord

from query_processing.adapters.vector_store.elasticsearch.elasticsearch_store import (
    ElasticsearchVectorStore,
)
from query_processing.domain.components.component_requests import SearchRequest
from query_processing.domain.errors import RagDependencyError

INDEX = "records_under_test"  # any name: the adapter must use the one it is given


def record(record_id: str, text: str = "t") -> VectorStoreRecord:
    """A record to write, its id in its metadata."""
    return VectorStoreRecord(text=text, metadata={"id": record_id})


# --- ElasticsearchVectorStore ----------------------------------------------


def es_hit(record_id: str, text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"_id": record_id, "_source": {"text": text, "metadata": metadata}}


class FakeLangChainStore:
    """Double for LangChain's ``AsyncElasticsearchStore``: scripted hits or a failure.

    ``client`` exposes only the elasticsearch-py coroutines the adapter uses;
    ``indices.create`` records each index created and makes it exist, and
    ``indices.get_mapping`` reports ``text_analyzer`` on the text field (``None``
    for a text field mapped without one).
    """

    query_field, vector_query_field = "text", "vector"  # LangChain's defaults

    def __init__(
        self,
        hits: Sequence[tuple[dict[str, Any], float]] = (),
        *,
        exists: bool = True,
        count: int = 0,
        lexical_hits: Sequence[dict[str, Any]] = (),
        failure: Exception | None = None,
        text_analyzer: str | None = None,
    ) -> None:
        self.hits, self.exists, self.count = list(hits), exists, count
        self.lexical_hits, self.failure = list(lexical_hits), failure
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.count_calls: list[str] = []
        self.lexical_calls: list[dict[str, Any]] = []
        self.closed = False
        self.text_analyzer = text_analyzer
        self.exists_calls: list[str] = []
        self.created: list[dict[str, Any]] = []
        indices = SimpleNamespace(
            exists=self._indices_exist,
            create=self._create_index,
            get_mapping=self._get_mapping,
        )
        self.client = SimpleNamespace(
            indices=indices, count=self._count_docs, search=self._search_text
        )

    async def _indices_exist(self, *, index: str) -> bool:
        self._maybe_fail()
        self.exists_calls.append(index)
        return self.exists

    async def _create_index(self, *, index: str, **body: Any) -> dict[str, Any]:
        self._maybe_fail()
        self.created.append({"index": index, **body})
        self.exists = True
        return {"acknowledged": True, "index": index}

    async def _get_mapping(self, *, index: str) -> dict[str, Any]:
        text: dict[str, Any] = {"type": "text"}
        if self.text_analyzer is not None:
            text["analyzer"] = self.text_analyzer
        return {index: {"mappings": {"properties": {"text": text}}}}

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
        self,
        embedding: list[float],
        k: int,
        filter: list[dict[str, Any]] | None = None,
        *,
        doc_builder: Callable[..., Any],
    ) -> list[tuple[Any, float]]:
        self._maybe_fail()
        self.search_calls.append({"embedding": embedding, "k": k, "filter": filter})
        return [(doc_builder(hit), score) for hit, score in self.hits[:k]]

    async def aclose(self) -> None:
        self.closed = True

    def _maybe_fail(self) -> None:
        if self.failure is not None:
            raise self.failure


def dense_strategy() -> AsyncDenseVectorStrategy:
    """The strategy ``open`` hands LangChain: its default, made explicit."""
    return AsyncDenseVectorStrategy(
        distance=DistanceMetric.COSINE, hybrid=False, rrf=False
    )


def plain_store(fake: FakeLangChainStore) -> ElasticsearchVectorStore:
    """The adapter with the default analyzer, as ``open`` builds it."""
    return ElasticsearchVectorStore(fake, INDEX, index_strategy=dense_strategy())


async def test_es_query_builds_scored_records_from_hits() -> None:
    fake = FakeLangChainStore(
        [
            (es_hit("p9", "nine", {"topic": "it"}), np.float32(0.75)),
            (es_hit("p4", "four", {}), 0.25),
        ]
    )
    hits = await plain_store(fake).query((0.5, 0.5), top_k=2)
    assert [(h.id, h.text, h.metadata) for h in hits] == [
        ("p9", "nine", {"topic": "it", "id": "p9"}),
        ("p4", "four", {"id": "p4"}),
    ]
    scores = [h.retrieval_score for h in hits]
    assert scores == [0.75, 0.25]
    assert scores == sorted(scores, reverse=True)
    assert all(type(h.retrieval_score) is float for h in hits)
    assert all(h.rerank_score is None for h in hits)
    assert fake.search_calls == [{"embedding": [0.5, 0.5], "k": 2, "filter": []}]


async def test_es_lexical_query_is_a_match_query_on_the_text_field() -> None:
    fake = FakeLangChainStore(
        lexical_hits=[
            {**es_hit("p7", "vacation days", {"topic": "hr"}), "_score": 3.5},
            {**es_hit("p2", "vacation", {}), "_score": 1.25},
        ]
    )
    hits = await plain_store(fake).lexical_query("vacation", 2)
    assert fake.lexical_calls == [
        {"index": INDEX, "query": {"match": {"text": "vacation"}}, "size": 2}
    ]
    assert [(h.id, h.text, h.metadata, h.retrieval_score) for h in hits] == [
        ("p7", "vacation days", {"topic": "hr", "id": "p7"}, 3.5),
        ("p2", "vacation", {"id": "p2"}, 1.25),
    ]
    assert all(type(h.retrieval_score) is float for h in hits)
    assert all(h.rerank_score is None for h in hits)


async def test_es_lexical_query_without_a_match_is_empty() -> None:
    fake = FakeLangChainStore()
    assert await plain_store(fake).lexical_query("x", 5) == []


TOPIC_HR = [MetadataCondition(attribute="topic", comparator="eq", value="hr")]
TOPIC_HR_CLAUSES = [{"term": {"metadata.topic.keyword": "hr"}}]
ACTIVE = MetadataCondition(attribute="active", comparator="eq", value=True)
ACTIVE_FIRST = [{"term": {"metadata.active": True}}, *TOPIC_HR_CLAUSES]


async def test_es_query_sends_the_conditions_as_the_vector_searchs_filter() -> None:
    fake = FakeLangChainStore()
    await plain_store(fake).query([0.5, 0.5], 2, TOPIC_HR)
    assert fake.search_calls == [
        {"embedding": [0.5, 0.5], "k": 2, "filter": TOPIC_HR_CLAUSES}
    ]


async def test_es_lexical_query_sends_the_conditions_in_its_bool_filter() -> None:
    fake = FakeLangChainStore()
    await plain_store(fake).lexical_query("vacation", 2, TOPIC_HR)
    query = {"match": {"text": "vacation"}}
    assert fake.lexical_calls == [
        {
            "index": INDEX,
            "query": {"bool": {"must": query, "filter": TOPIC_HR_CLAUSES}},
            "size": 2,
        }
    ]


@pytest.mark.parametrize(
    ("prefilter", "clauses"), [((), TOPIC_HR_CLAUSES), ((ACTIVE,), ACTIVE_FIRST)]
)
async def test_es_hybrid_search_passes_the_conditions_to_both_searches(
    prefilter: tuple[MetadataCondition, ...], clauses: list[dict[str, Any]]
) -> None:
    fake = FakeLangChainStore()
    store = plain_store(fake)
    store.prefilter = prefilter
    request = SearchRequest(
        embedding=[0.5, 0.5],
        text="vacation",
        top_k=2,
        retrieval_mode="hybrid",
        rrf_k=60,
        metadata_filter=TOPIC_HR,
    )
    await store.ainvoke(request)
    assert [call["filter"] for call in fake.search_calls] == [clauses]
    [lexical] = fake.lexical_calls
    assert lexical["query"]["bool"]["filter"] == clauses


@pytest.mark.parametrize(("exists", "expected"), [(False, 0), (True, 17)])
async def test_es_count_is_zero_without_index(exists: bool, expected: int) -> None:
    fake = FakeLangChainStore(exists=exists, count=17)
    assert await plain_store(fake).count() == expected
    assert fake.count_calls == ([INDEX] if exists else [])


@pytest.mark.parametrize(
    ("exists", "expected"),
    [
        (False, (False, f"0 records in {INDEX}")),
        (True, (True, f"17 records in {INDEX}")),
    ],
)
async def test_es_ready_reports_the_record_count_and_index(
    exists: bool, expected: tuple[bool, str]
) -> None:
    fake = FakeLangChainStore(exists=exists, count=17)
    assert await plain_store(fake).ready() == expected


async def test_es_aclose_closes_the_underlying_store() -> None:
    fake = FakeLangChainStore()
    await plain_store(fake).aclose()
    assert fake.closed


StoreOperation = Callable[[ElasticsearchVectorStore], Awaitable[Any]]


@pytest.mark.parametrize(
    "operation",
    [
        lambda store: store.upsert([record("p1", "alpha")], [[0.1]]),
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
    store = plain_store(FakeLangChainStore(failure=boom))
    with pytest.raises(RagDependencyError) as info:
        await operation(store)
    assert info.value.component == "vector_store"
    assert info.value.target == INDEX
    assert info.value.kind == "unavailable"
    assert info.value.detail == "ConnectionError: es refused"
    assert info.value.__cause__ is boom


# --- the analyzer of the records' text -------------------------------------


def french_store(fake: FakeLangChainStore) -> ElasticsearchVectorStore:
    return ElasticsearchVectorStore(
        fake, INDEX, "french", index_strategy=dense_strategy()
    )


async def test_es_a_new_index_is_created_with_langchains_mapping_and_the_analyzer() -> (
    None
):
    fake = FakeLangChainStore(exists=False)
    store = french_store(fake)
    await store.upsert([record("p1", "la prime")], [[0.1, 0.2, 0.3]])
    await store.upsert(
        [record("p2", "le sinistre")],
        [[0.3, 0.2, 0.1]],
    )
    assert fake.created == [
        {
            "index": INDEX,
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": 3,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "text": {"type": "text", "analyzer": "french"},
                }
            },
            "settings": {},
        }
    ]
    assert fake.exists_calls == [INDEX]  # checked once, not on every upsert
    assert [c["ids"] for c in fake.add_calls] == [["p1"], ["p2"]]


async def test_es_an_existing_index_is_left_alone() -> None:
    fake = FakeLangChainStore(exists=True)
    await french_store(fake).upsert([record("p1")], [[0.1]])
    assert fake.created == []
    assert len(fake.add_calls) == 1


@pytest.mark.parametrize("analyzer", ["", "standard"])
async def test_es_the_standard_analyzer_leaves_the_text_field_to_dynamic_mapping(
    analyzer: str,
) -> None:
    fake = FakeLangChainStore(exists=False)
    store = ElasticsearchVectorStore(
        fake, INDEX, analyzer, index_strategy=dense_strategy()
    )
    await store.upsert([record("p1")], [[0.1, 0.2]])
    vector = {"type": "dense_vector", "dims": 2, "index": True, "similarity": "cosine"}
    assert fake.created == [
        {"index": INDEX, "mappings": {"properties": {"vector": vector}}, "settings": {}}
    ]
    assert len(fake.add_calls) == 1


async def test_es_a_failed_index_creation_is_a_vector_store_error() -> None:
    boom = ConnectionError("es refused")
    fake = FakeLangChainStore(exists=False, failure=boom)
    with pytest.raises(RagDependencyError) as info:
        await french_store(fake).upsert([record("p1")], [[0.1]])
    assert (info.value.component, info.value.target) == ("vector_store", INDEX)
    assert info.value.__cause__ is boom


@pytest.mark.parametrize(
    ("index_analyzer", "expected"),
    [
        ("french", f"17 records in {INDEX}"),
        (
            None,
            f"17 records in {INDEX}; its text analyzer is 'standard', not the "
            "configured 'french': an index keeps the analyzer it was created with",
        ),
    ],
    ids=["same", "different"],
)
async def test_es_ready_reports_an_index_analyzer_that_differs(
    index_analyzer: str | None, expected: str
) -> None:
    fake = FakeLangChainStore(exists=True, count=17, text_analyzer=index_analyzer)
    assert await french_store(fake).ready() == (True, expected)


async def test_es_open_hands_langchain_the_strategy_the_index_is_built_from() -> None:
    # A contract check on the installed LangChain: no request is sent.
    store = ElasticsearchVectorStore.open(
        "http://localhost:9200", index=INDEX, text_analyzer="french"
    )
    try:
        langchain_store = store._store
        assert langchain_store._store.retrieval_strategy is store._index_strategy
        assert (langchain_store.query_field, langchain_store.vector_query_field) == (
            "text",
            "vector",
        )
    finally:
        await store.aclose()


# --- the first write into a missing index, through the installed LangChain -----


class FakeHeadResponse:
    """What ``indices.exists`` answers: truthy for a 2xx, and a ``meta.status``."""

    def __init__(self, status: int) -> None:
        self.meta = SimpleNamespace(status=status)

    def __bool__(self) -> bool:
        return 200 <= self.meta.status < 300


class FakeAsyncElasticsearch:
    """Double for the ``AsyncElasticsearch`` client under LangChain's real store.

    It answers what LangChain's index creation and elasticsearch-py's
    ``async_bulk`` call: ``options``, ``indices.exists`` / ``create``, ``bulk`` and
    the transport's JSON serializer, and the vector search's ``search``. Created
    indices, bulk-indexed ids and search bodies are recorded.
    """

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.indexed_ids: list[str] = []
        self.searches: list[dict[str, Any]] = []
        self.indices = SimpleNamespace(exists=self._exists, create=self._create)
        serializers = SimpleNamespace(get_serializer=lambda _mime: JsonSerializer())
        self.transport = SimpleNamespace(serializers=serializers)

    def options(self, **_kwargs: Any) -> FakeAsyncElasticsearch:
        return self

    async def _exists(self, *, index: str) -> FakeHeadResponse:
        made = any(c["index"] == index for c in self.created)
        return FakeHeadResponse(200 if made else 404)

    async def _create(self, *, index: str, **body: Any) -> dict[str, Any]:
        self.created.append({"index": index, **body})
        return {"acknowledged": True}

    async def bulk(self, *, operations: list[bytes], **_kwargs: Any) -> Any:
        headers = [json.loads(op) for op in operations[::2]]
        self.indexed_ids += [h["index"]["_id"] for h in headers]
        items = [{"index": {"_id": h["index"]["_id"], "status": 201}} for h in headers]
        return SimpleNamespace(body={"errors": False, "items": items})

    async def search(self, **body: Any) -> dict[str, Any]:
        self.searches.append(body)
        return {"hits": {"hits": []}}

    async def close(self) -> None:
        pass


def langchain_store_over(client: FakeAsyncElasticsearch) -> ElasticsearchVectorStore:
    """The adapter on LangChain's real store, built as ``open`` builds it."""
    strategy = dense_strategy()
    store = AsyncElasticsearchStore(INDEX, client=client, strategy=strategy)
    return ElasticsearchVectorStore(store, INDEX, index_strategy=strategy)


async def test_es_first_upsert_into_a_missing_index_works_with_the_default_analyzer() -> (
    None
):
    # Regression: LangChain's store has no embedding and no num_dimensions, so its
    # own index creation raised "retrieval strategy requires embeddings; either
    # embedding_service or num_dimensions need to be specified".
    client = FakeAsyncElasticsearch()
    store = langchain_store_over(client)
    await store.upsert([record("p1", "la prime")], [[0.1, 0.2, 0.3]])
    assert client.indexed_ids == ["p1"]
    vector = client.created[0]["mappings"]["properties"]["vector"]
    assert (vector["type"], vector["dims"]) == ("dense_vector", 3)


async def test_es_langchain_puts_the_conditions_inside_the_knn_clause() -> None:
    # A contract check on the installed LangChain: the clauses are the kNN
    # search's own filter, applied during the search, not a filter after it.
    client = FakeAsyncElasticsearch()
    await langchain_store_over(client).query([0.1, 0.2], 3, TOPIC_HR)
    [body] = client.searches
    assert body["knn"]["filter"] == TOPIC_HR_CLAUSES
    assert "query" not in body
