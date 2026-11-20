"""``ElasticsearchVectorStore`` and a record's id: written with it, read with the _id.

A record keeps its id in its metadata; Elasticsearch holds a document under its
``_id``. Over the fake LangChain store of ``test_elasticsearch_store``: no
Elasticsearch.
"""

from __future__ import annotations

from typing import Any

import pytest
from query_processing_core.retrieval.vector_store_record import VectorStoreRecord
from test_elasticsearch_store import FakeLangChainStore, es_hit, plain_store


async def test_es_upsert_passes_text_vector_pairs_metadatas_and_ids() -> None:
    fake = FakeLangChainStore()
    passages = [
        VectorStoreRecord(text="alpha", metadata={"id": "p1", "topic": "hr"}),
        VectorStoreRecord(text="beta", metadata={"id": "p2"}),
    ]
    store = plain_store(fake)
    await store.upsert(passages, [(0.1, 0.2), [0.3, 0.4]])
    await store.upsert([], [])  # nothing to write: no call at all
    assert fake.add_calls == [
        {
            "pairs": [("alpha", [0.1, 0.2]), ("beta", [0.3, 0.4])],
            "metadatas": [{"id": "p1", "topic": "hr"}, {"id": "p2"}],
            "ids": ["p1", "p2"],
        }
    ]


# Stored metadata as an index ingested before holds it (no id), and as one
# whose id is not the document's: a record read has Elasticsearch's _id.
STORED_METADATA = [
    ({"topic": "hr"}, {"topic": "hr", "id": "p9"}),
    ({"id": "stale", "topic": "hr"}, {"id": "p9", "topic": "hr"}),
]


@pytest.mark.parametrize(("stored", "read"), STORED_METADATA)
async def test_es_a_record_found_by_vector_has_the_id_of_its_document(
    stored: dict[str, Any], read: dict[str, Any]
) -> None:
    fake = FakeLangChainStore([(es_hit("p9", "nine", stored), 0.5)])
    [hit] = await plain_store(fake).query((0.5, 0.5), top_k=1)
    assert (hit.id, hit.metadata) == ("p9", read)


@pytest.mark.parametrize(("stored", "read"), STORED_METADATA)
async def test_es_a_record_found_by_words_has_the_id_of_its_document(
    stored: dict[str, Any], read: dict[str, Any]
) -> None:
    found = {**es_hit("p9", "nine", stored), "_score": 1.0}
    [hit] = await plain_store(FakeLangChainStore(lexical_hits=[found])).lexical_query(
        "nine", 1
    )
    assert (hit.id, hit.metadata) == ("p9", read)
