"""The vector-store record, doclang's, and the record with its two scores.

doclang's own tests build chunks with its ingestion package, which this
repository does not install: the chunk here is built by hand from doclang's
core, one version, one page and one block holding one text element.
"""

from __future__ import annotations

from typing import Any, get_args

import pytest
from doclang_domain.core.adapters.implementations import vector_store as doclang
from doclang_domain.core.chunking import Chunk, ChunkBlock
from doclang_domain.core.document.elements.text import Text
from doclang_domain.core.shared import DocumentHash, DocumentVersionRef, chunk_id_for
from pydantic import ValidationError

from query_processing_core.retrieval.metadata_filter import MetadataValue
from query_processing_core.retrieval.vector_store_record import (
    ScoredVectorStoreRecord,
    VectorStoreRecord,
)

SENTENCE = "Leave is booked a week ahead."


def handbook_chunk() -> Chunk:
    """The only chunk of a one-page document's first version, one sentence long."""
    version = DocumentVersionRef(
        document_id="doc-handbook",
        version_no=1,
        document_hash=DocumentHash(value="ab12", hash_function="sha256sum"),
    )
    text = Text.model_validate(
        {"id": "text-1", "content": [{"kind": "span", "text": SENTENCE}]}
    )
    return Chunk(
        id=chunk_id_for(version, 0),
        version_ref=version,
        sequence=0,
        chunk_count=1,
        blocks=(ChunkBlock(page_no=1, block_index=0, element=text),),
        page_count=1,
    )


def scored(record: VectorStoreRecord, **scores: float) -> ScoredVectorStoreRecord:
    """``record`` with ``scores``, as a search and a rerank give them."""
    return ScoredVectorStoreRecord(**record.model_dump(), **scores)


def test_the_core_hands_on_doclangs_own_record() -> None:
    assert VectorStoreRecord is doclang.VectorStoreRecord
    assert issubclass(ScoredVectorStoreRecord, doclang.VectorStoreRecord)


def test_a_record_made_from_a_chunk_keeps_its_chunk_through_json() -> None:
    chunk = handbook_chunk()
    record = scored(
        doclang.to_vector_store(chunk), retrieval_score=0.82, rerank_score=3.5
    )

    back = ScoredVectorStoreRecord.model_validate_json(record.model_dump_json())

    assert back == record
    assert (back.retrieval_score, back.rerank_score) == (0.82, 3.5)
    assert back.id == chunk.id == "doc-handbook@v1#chunk-0"
    assert SENTENCE in back.text
    assert doclang.from_vector_store(back) == chunk


def test_a_record_with_no_chunk_is_valid_and_doclang_refuses_to_rebuild_one() -> None:
    record = ScoredVectorStoreRecord(
        text=SENTENCE, metadata={"id": "handbook-7", "sequence": 7}, retrieval_score=1.0
    )

    assert record.id == "handbook-7"
    assert record.rerank_score is None
    with pytest.raises(ValueError, match="the record has no 'chunk' payload"):
        doclang.from_vector_store(record)


@pytest.mark.parametrize(
    ("fields", "refused"),
    [
        ({"text": "t", "metadata": {"id": "a"}}, "retrieval_score"),
        (
            {"text": "t", "metadata": {"id": "a"}, "retrieval_score": 1.0, "score": 2},
            "Extra inputs are not permitted",
        ),
    ],
    ids=["without-its-retrieval-score", "with-an-unknown-key"],
)
def test_a_scored_record_refuses_what_the_record_refuses(
    fields: dict[str, Any], refused: str
) -> None:
    with pytest.raises(ValidationError, match=refused):
        ScoredVectorStoreRecord.model_validate(fields)


def test_a_scored_record_is_frozen() -> None:
    # Typed as Any: the checkers already refuse the assignment this test makes.
    record: Any = ScoredVectorStoreRecord(
        text="t", metadata={"id": "a"}, retrieval_score=1.0
    )

    with pytest.raises(ValidationError, match="frozen"):
        record.rerank_score = 2.0


def test_a_metadata_value_has_doclangs_scalar_type() -> None:
    # The metadata filter keeps its own name for the type, and conditions on a
    # record's metadata must take exactly the values a record holds.
    assert set(get_args(MetadataValue)) == set(get_args(doclang.Scalar))
