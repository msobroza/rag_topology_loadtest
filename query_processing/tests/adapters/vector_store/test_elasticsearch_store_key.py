"""The Elasticsearch store's API key, from its key source, over a fake LangChain store.

The store opens its client with the key its source gives, and with none when it
has no source. When Elasticsearch refuses that key (its 401), the store tells
its source, opens a new client with the key the source then gives and makes the
call once more: one further attempt, for every call that reaches Elasticsearch.
A replaced client is closed with the store, so a call in flight on it ends. No error holds a key. Basic authentication written in
the URL is the client's own, as before.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import AuthenticationException, AuthorizationException
from test_elasticsearch_store import INDEX, FakeLangChainStore, dense_strategy, es_hit

from model_api_fakes import FakeKeySource
from query_processing.adapters.vector_store.elasticsearch.elasticsearch_store import (
    ElasticsearchVectorStore,
)
from query_processing.domain.errors import RagDependencyError
from query_processing.domain.retrieval.passages import Passage

OLD_KEY, MID_KEY, NEW_KEY = "es-key-old-4711", "es-key-mid-2342", "es-key-new-0815"
StoreOperation = Callable[[ElasticsearchVectorStore], Awaitable[Any]]
OPERATIONS = pytest.mark.parametrize(
    "operation",
    [
        lambda store: store.upsert([Passage(id="p1", text="alpha")], [[0.1]]),
        lambda store: store.query([0.1], top_k=1),
        lambda store: store.lexical_query("alpha", top_k=1),
        lambda store: store.ready(),
    ],
    ids=["upsert", "query", "lexical_query", "ready"],
)


def es_refusal(status: int, key: str) -> Exception:
    """What Elasticsearch's client raises for ``status``, quoting the key it sent."""
    node = NodeConfig("http", "localhost", 9200)
    meta = ApiResponseMeta(status, "1.1", HttpHeaders(), 0.0, node)
    error = {401: AuthenticationException, 403: AuthorizationException}[status]
    return error(f"key {key} refused", meta=meta, body={"key": key})


class ClosableLangChainStore(FakeLangChainStore):
    """The fake LangChain store, failing any use after its close and counting closes."""

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.closes = 0

    async def aclose(self) -> None:
        await super().aclose()
        self.closes += 1

    def _maybe_fail(self) -> None:
        if self.closed:
            raise ConnectionError("the client is closed")
        super()._maybe_fail()


class InFlightLangChainStore(ClosableLangChainStore):
    """A client whose vector search is in flight until released; its text search is
    refused for its key (a 401)."""

    def __init__(self, key: str) -> None:
        super().__init__(hits=[(es_hit("p1", "alpha", {}), 0.5)], exists=True)
        self.refusal, self.released = es_refusal(401, key), asyncio.Event()
        self.search_started = asyncio.Event()

    async def _search_text(
        self, *, index: str, query: dict[str, Any], size: int
    ) -> dict[str, Any]:
        raise self.refusal

    async def asimilarity_search_by_vector_with_relevance_scores(
        self, embedding: list[float], k: int, **options: Any
    ) -> list[tuple[Any, float]]:
        self.search_started.set()
        await self.released.wait()  # the answer comes after the other call's refusal
        return await super().asimilarity_search_by_vector_with_relevance_scores(
            embedding, k, **options
        )


class FakeConnector:
    """Builds a fake LangChain store for each key; a refused key's store gets ``status``.

    It records the key each store was built with, in order, and the stores.
    """

    def __init__(self, *refused: str, status: int = 401) -> None:
        self.refused, self.status = set(refused), status
        self.keys: list[str | None] = []
        self.stores: list[ClosableLangChainStore] = []

    def __call__(self, key: str | None) -> ClosableLangChainStore:
        failure = (
            es_refusal(self.status, str(key)) if str(key) in self.refused else None
        )
        store = ClosableLangChainStore(exists=True, count=3, failure=failure)
        self.keys.append(key)
        self.stores.append(store)
        return store


async def store_over(
    connect: FakeConnector, source: FakeKeySource | None
) -> ElasticsearchVectorStore:
    return await ElasticsearchVectorStore.open_with(
        connect, INDEX, index_strategy=dense_strategy(), key_source=source
    )


@pytest.mark.parametrize(
    ("source", "key"), [(FakeKeySource(OLD_KEY), OLD_KEY), (None, None)]
)
async def test_es_the_store_opens_with_the_key_its_source_gives_or_none(
    source: FakeKeySource | None, key: str | None
) -> None:
    connect = FakeConnector()
    await store_over(connect, source)
    assert connect.keys == [key]


@OPERATIONS
async def test_es_a_refused_key_opens_a_new_client_with_the_new_key_and_calls_again(
    operation: StoreOperation,
) -> None:
    source, connect = FakeKeySource(OLD_KEY, NEW_KEY), FakeConnector(OLD_KEY)
    store = await store_over(connect, source)

    await operation(store)

    assert source.refusals == [OLD_KEY]
    assert connect.keys == [OLD_KEY, NEW_KEY]
    old, new = connect.stores
    assert not old.closed  # the old one is closed with the store
    assert not new.closed
    assert new.exists_calls or new.search_calls or new.lexical_calls


async def test_es_the_new_client_serves_the_next_calls_and_is_closed_with_the_store() -> (
    None
):
    connect = FakeConnector(OLD_KEY)
    store = await store_over(connect, FakeKeySource(OLD_KEY, NEW_KEY))
    await store.query([0.1], top_k=1)
    await store.lexical_query("alpha", top_k=1)
    await store.aclose()
    assert connect.keys == [OLD_KEY, NEW_KEY]
    assert connect.stores[1].closed


async def test_es_a_call_in_flight_on_the_replaced_client_still_gets_its_answer() -> (
    None
):
    old = InFlightLangChainStore(OLD_KEY)
    connect = FakeConnector()
    keyed = iter([old, None])

    def connect_old_then_new(key: str | None) -> ClosableLangChainStore:
        return next(keyed) or connect(key)

    store = await ElasticsearchVectorStore.open_with(
        connect_old_then_new,
        INDEX,
        index_strategy=dense_strategy(),
        key_source=FakeKeySource(OLD_KEY, NEW_KEY),
    )
    in_flight = asyncio.create_task(store.query([0.1], top_k=1))
    await old.search_started.wait()

    await store.lexical_query("alpha", top_k=1)  # refused: a new client replaces it
    old.released.set()

    [hit] = await in_flight
    assert hit.id == "p1"
    assert connect.keys == [NEW_KEY]


async def test_es_aclose_closes_every_client_opened_the_replaced_ones_included() -> (
    None
):
    connect = FakeConnector(OLD_KEY, MID_KEY)
    store = await store_over(connect, FakeKeySource(OLD_KEY, MID_KEY, NEW_KEY))
    with pytest.raises(RagDependencyError):
        await store.count()  # OLD_KEY, then MID_KEY, refused
    await store.count()  # MID_KEY refused, NEW_KEY taken

    await store.aclose()

    assert connect.keys == [OLD_KEY, MID_KEY, NEW_KEY]
    assert [client.closes for client in connect.stores] == [1, 1, 1]


@OPERATIONS
async def test_es_a_second_refusal_is_the_stores_error_naming_the_source_not_the_key(
    operation: StoreOperation,
) -> None:
    source = FakeKeySource(OLD_KEY, NEW_KEY)
    connect = FakeConnector(OLD_KEY, NEW_KEY)
    store = await store_over(connect, source)

    with pytest.raises(RagDependencyError) as info:
        await operation(store)

    assert connect.keys == [OLD_KEY, NEW_KEY]  # one further attempt, no more
    assert (info.value.component, info.value.target) == ("vector_store", INDEX)
    assert source.described() in str(info.value)
    assert "AuthenticationException" in str(info.value)
    assert OLD_KEY not in str(info.value)
    assert NEW_KEY not in str(info.value)


async def test_es_a_403_fails_at_once_and_keeps_its_key() -> None:
    source = FakeKeySource(OLD_KEY, NEW_KEY)
    connect = FakeConnector(OLD_KEY, status=403)
    store = await store_over(connect, source)
    with pytest.raises(RagDependencyError) as info:
        await store.query([0.1], top_k=1)
    assert "AuthorizationException" in info.value.detail
    assert OLD_KEY not in str(info.value)
    assert (source.refusals, connect.keys) == ([], [OLD_KEY])


async def test_es_a_401_without_a_key_source_fails_at_once() -> None:
    connect = FakeConnector("None")  # the key the fake quotes for no key
    store = await store_over(connect, None)
    with pytest.raises(RagDependencyError) as info:
        await store.count()
    assert "AuthenticationException" in info.value.detail
    assert connect.keys == [None]


async def test_es_open_gives_langchains_client_the_key_as_an_elasticsearch_api_key() -> (
    None
):
    # A contract check on the installed client: no request is sent.
    store = await ElasticsearchVectorStore.open(
        "http://localhost:9200", index=INDEX, key_source=FakeKeySource(OLD_KEY)
    )
    try:
        authorization = store._store.client._headers["authorization"]
        assert authorization.startswith("ApiKey ")
    finally:
        await store.aclose()


async def test_es_basic_authentication_in_the_url_reaches_the_client_without_a_source() -> (
    None
):
    # A contract check on the installed client: no request is sent.
    store = await ElasticsearchVectorStore.open(
        "http://user:password@localhost:9200", index=INDEX
    )
    try:
        client = store._store.client
        [node] = client.transport.node_pool.all()
        assert node.config.headers["authorization"].startswith("Basic ")
        assert "authorization" not in client._headers
    finally:
        await store.aclose()
