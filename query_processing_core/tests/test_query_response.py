"""The body of a ``/query`` or ``/retrieve`` response, as a client reads it."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from query_processing_core.query_response import TOTAL_TIMING_KEY, QueryResponse

# A body as the workflow app writes it: a record made from a chunk, its chunk
# a JSON text under "chunk", and a record of the handbook, which holds none.
BODY = {
    "request_id": "req-1",
    "topology": None,
    "run_label": "fake",
    "question": "Vacation days?",
    "mode": "query",
    "answer": "Twenty days.",
    "records": [
        {
            "text": "Leave is booked a week ahead.",
            "metadata": {"id": "doc@v1#chunk-0", "chunk": "{}", "sequence": 0},
            "retrieval_score": 0.82,
            "rerank_score": 3.5,
        },
        {
            "text": "Twenty days a year.",
            "metadata": {"id": "doc-0001:0", "sequence": 0},
            "retrieval_score": 0.5,
            "rerank_score": None,
        },
    ],
    "timings_ms": {"embed": 1.0, TOTAL_TIMING_KEY: 2.0},
    "tokens": {},
    "metadata_filter": [],
}


def test_a_body_is_read_with_its_records_as_they_are_and_written_back_alike() -> None:
    response = QueryResponse.model_validate_json(json.dumps(BODY))

    assert [record.id for record in response.records] == [
        "doc@v1#chunk-0",
        "doc-0001:0",
    ]
    assert response.records[0].metadata["chunk"] == "{}"
    assert json.loads(response.model_dump_json()) == BODY


def test_a_body_of_another_mode_is_refused() -> None:
    with pytest.raises(ValidationError, match="'query' or 'retrieve'"):
        QueryResponse.model_validate({**BODY, "mode": "rerank"})
