"""The model server's side of the API formats: its answers, and the rerank round trip.

Everything here is pure: no HTTP, no server. ``test_model_server.py`` tests the
wire on top of it.
"""

from __future__ import annotations

import pytest

from rag_load_test.adapters.model_api.cohere_rerank_format import CohereRerankFormat
from rag_load_test.adapters.model_api.model_server_contract import (
    UNKNOWN_MODEL_ERROR,
    candidates_from_rerank,
    embeddings_response_for,
    rerank_response_for,
)
from rag_load_test.adapters.model_api.openai_embeddings_format import (
    OpenAIEmbeddingsFormat,
)
from rag_load_test.domain.standard_formats import CohereRerankRequest


def test_unknown_model_error_is_the_ovms_text() -> None:
    expected = "Mediapipe graph definition with requested name is not found"
    assert expected == UNKNOWN_MODEL_ERROR


def test_embeddings_response_for_numbers_rows_in_input_order() -> None:
    response = embeddings_response_for("e", [[0.1, 0.2], [0.3, 0.4]])
    assert response.model_dump() == {
        "object": "list",
        "model": "e",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
        ],
    }
    body = response.model_dump()
    assert OpenAIEmbeddingsFormat().vectors(body) == [[0.1, 0.2], [0.3, 0.4]]


def test_candidates_from_rerank_carry_their_document_index_as_id() -> None:
    request = CohereRerankRequest(model="r", query="q", documents=["a", "b"])
    candidates = candidates_from_rerank(request)
    assert [(p.id, p.text) for p in candidates] == [("0", "a"), ("1", "b")]
    assert all(p.rerank_score is None for p in candidates)
    assert all(p.retrieval_score == 0.0 for p in candidates)


def test_rerank_response_for_names_documents_by_index() -> None:
    request = CohereRerankRequest(model="r", query="q", documents=["a", "b", "c"])
    candidates = candidates_from_rerank(request)
    ranked = [
        candidates[2].model_copy(update={"rerank_score": 0.9}),
        candidates[0].model_copy(update={"rerank_score": 0.4}),
    ]
    assert rerank_response_for(ranked).model_dump() == {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.4},
        ]
    }


def test_rerank_round_trip_maps_back_to_the_documents_sent() -> None:
    api_format = CohereRerankFormat()
    body = api_format.request_body("r", "q", ["a", "b", "c"], top_n=2)
    candidates = candidates_from_rerank(CohereRerankRequest.model_validate(body))
    ranked = [candidates[1].model_copy(update={"rerank_score": 0.8})]
    answer = rerank_response_for(ranked).model_dump()
    assert api_format.scores(answer) == {1: 0.8}


def test_rerank_response_for_rejects_an_unscored_passage() -> None:
    [unscored] = candidates_from_rerank(
        CohereRerankRequest(model="r", query="q", documents=["a"])
    )
    with pytest.raises(ValueError, match="'0' has no rerank_score"):
        rerank_response_for([unscored])
