"""The cohere rerank format: a pure data transformation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_load_test.adapters.reranker.api.cohere_rerank_format import CohereRerankFormat


def test_rerank_request_has_the_cohere_shape() -> None:
    body = CohereRerankFormat().request_body("r", "q", ("a", "b"), 1)
    assert body == {"model": "r", "query": "q", "documents": ["a", "b"], "top_n": 1}


def test_rerank_scores_keep_the_answers_order() -> None:
    results = [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.1},
    ]
    scores = CohereRerankFormat().scores({"id": "x", "results": results})
    assert list(scores.items()) == [(2, 0.9), (0, 0.1)]


@pytest.mark.parametrize(
    "body", [{"data": []}, {"results": [{"index": "x"}]}], ids=["no-results", "bad-row"]
)
def test_rerank_malformed_bodies_fail_validation(body: object) -> None:
    with pytest.raises(ValidationError):
        CohereRerankFormat().scores(body)
