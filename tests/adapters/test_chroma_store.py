"""``ChromaVectorStore`` on a real embedded Chroma in ``tmp_path``: no server."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from rag_load_test.adapters.vector_store.chroma.chroma_store import ChromaVectorStore
from rag_load_test.domain.errors import RagDependencyError
from rag_load_test.domain.passages import Passage

COLLECTION = "passages_under_test"  # any name: the adapter must use the one given


async def open_store(path: Path) -> Any:
    return await ChromaVectorStore.open(url="", path=str(path), collection=COLLECTION)


PASSAGES = [
    Passage(id="p1", text="vacation days accrue monthly", metadata={"topic": "hr"}),
    Passage(id="p2", text="laptops are replaced every three years"),
    Passage(id="p3", text="vacation vacation policy", metadata={"n": 3}),
]
VECTORS = [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]


# --- embedded store --------------------------------------------------------


async def test_chroma_upsert_stores_and_overwrites_by_id(tmp_path: Path) -> None:
    store = await open_store(tmp_path)
    await store.upsert([], [])  # nothing to write: no error
    await store.upsert(PASSAGES, VECTORS)
    await store.upsert([Passage(id="p2", text="laptops last four years")], [[0, 1]])
    assert await store.count() == 3
    hits = await store.query([0.0, 1.0], top_k=1)
    assert [(h.id, h.text, h.metadata) for h in hits] == [
        ("p2", "laptops last four years", {})
    ]


async def test_chroma_upsert_replaces_metadata_rather_than_merging(
    tmp_path: Path,
) -> None:
    store = await open_store(tmp_path)
    await store.upsert(PASSAGES, VECTORS)
    await store.upsert([Passage(id="p1", text="vacation days", metadata={})], [[1, 0]])
    [hit] = await store.query([1.0, 0.0], top_k=1)
    assert (hit.id, hit.text, hit.metadata) == ("p1", "vacation days", {})


async def test_chroma_query_is_nearest_first_with_cosine_similarity(
    tmp_path: Path,
) -> None:
    store = await open_store(tmp_path)
    await store.upsert(PASSAGES, VECTORS)
    hits = await store.query([1.0, 0.0], top_k=3)
    assert [h.id for h in hits] == ["p1", "p3", "p2"]
    scores = [h.retrieval_score for h in hits]
    assert scores == pytest.approx([1.0, 0.6, 0.0], abs=1e-5)
    assert hits[0].metadata == {"topic": "hr"}
    assert hits[1].metadata == {"n": 3}
    assert all(h.rerank_score is None for h in hits)


async def test_chroma_ready_reports_count_and_collection(tmp_path: Path) -> None:
    store = await open_store(tmp_path)
    assert await store.ready() == (False, f"0 passages in {COLLECTION}")
    await store.upsert(PASSAGES, VECTORS)
    assert await store.ready() == (True, f"3 passages in {COLLECTION}")


async def test_chroma_failures_surface_as_vector_store_errors(
    tmp_path: Path,
) -> None:
    store = await open_store(tmp_path)
    await store.upsert(PASSAGES, VECTORS)
    with pytest.raises(RagDependencyError) as info:
        await store.query([1.0, 0.0, 0.0], top_k=1)  # wrong dimension
    assert info.value.component == "vector_store"
    assert info.value.target == COLLECTION
    assert info.value.detail.startswith("InvalidArgumentError: ")
    assert info.value.__cause__ is not None


async def test_chroma_open_maps_a_bad_collection_name(tmp_path: Path) -> None:
    with pytest.raises(RagDependencyError) as info:
        await ChromaVectorStore.open(url="", path=str(tmp_path), collection="no!")
    assert info.value.target == "no!"


# --- server branch ---------------------------------------------------------


class FakeChromaCollection:
    """Double for a Chroma collection: only its name is read when opening."""

    name = COLLECTION


class FakeChromaClient:
    """Double for a Chroma client: records the collections asked for."""

    def __init__(self, collection: FakeChromaCollection) -> None:
        self.collection = collection
        self.collections: list[str] = []

    def get_or_create_collection(self, name: str, **_: Any) -> FakeChromaCollection:
        self.collections.append(name)
        return self.collection


class FakeChromaModule:
    """Double for the ``chromadb`` module: records how ``HttpClient`` was built."""

    def __init__(self) -> None:
        self.http_calls: list[dict[str, Any]] = []
        self.client = FakeChromaClient(FakeChromaCollection())

    def Settings(self, **kwargs: Any) -> dict[str, Any]:  # noqa: N802
        return kwargs

    def HttpClient(self, **kwargs: Any) -> FakeChromaClient:  # noqa: N802
        self.http_calls.append(kwargs)
        return self.client

    def PersistentClient(self, **_: Any) -> FakeChromaClient:  # noqa: N802
        raise AssertionError("a server URL must not open the embedded store")


@pytest.mark.parametrize(
    ("url", "host", "port", "ssl"),
    [
        ("http://chroma.internal:8123", "chroma.internal", 8123, False),
        ("https://chroma.example.com", "chroma.example.com", 443, True),
        ("http://localhost", "localhost", 80, False),
    ],
)
async def test_chroma_open_builds_an_http_client_from_the_url(
    monkeypatch: pytest.MonkeyPatch, url: str, host: str, port: int, ssl: bool
) -> None:
    fake = FakeChromaModule()
    monkeypatch.setitem(sys.modules, "chromadb", fake)
    store = await ChromaVectorStore.open(url=url, path="unused", collection=COLLECTION)
    assert isinstance(store, ChromaVectorStore)
    [call] = fake.http_calls
    assert (call["host"], call["port"], call["ssl"]) == (host, port, ssl)
    assert fake.client.collections == [COLLECTION]
