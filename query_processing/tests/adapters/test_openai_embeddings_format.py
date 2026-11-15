"""The openai embeddings format: a pure data transformation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_load_test.adapters.embedder.api.openai_embeddings_format import (
    OpenAIEmbeddingsFormat,
)


def test_embeddings_request_sends_input_as_a_list() -> None:
    body = OpenAIEmbeddingsFormat().request_body("m", ("only",))
    assert body == {"model": "m", "input": ["only"]}


def test_embeddings_vectors_come_back_in_request_order() -> None:
    rows = [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]
    body = {"object": "list", "data": rows, "usage": {"total_tokens": 2}}
    assert OpenAIEmbeddingsFormat().vectors(body) == [[1.0], [2.0]]


@pytest.mark.parametrize(
    "body",
    [
        {"object": "list"},
        {"data": [{"index": 0}]},
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 2, "embedding": [1.0]}]},
        [1, 2],
    ],
    ids=["no-data", "no-embedding", "gap-in-indexes", "not-an-object"],
)
def test_embeddings_malformed_bodies_fail_validation(body: object) -> None:
    with pytest.raises(ValidationError):
        OpenAIEmbeddingsFormat().vectors(body)
