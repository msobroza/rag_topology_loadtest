"""Unit tests for ``rag_load_test.v3_contract``: paths, shapes and index bookkeeping.

Everything here is pure: no HTTP, no server. The adapters (``test_ovms.py``) and
the FastAPI model server (``test_model_server.py``) test the wire on top of it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_load_test.models import ScoredPassage
from rag_load_test.v3_contract import (
    EMBEDDINGS_PATH,
    RERANK_PATH,
    UNKNOWN_MODEL_ERROR,
    EmbeddingsRequest,
    EmbeddingsResponse,
    RerankRequest,
    RerankResponse,
    candidates_from_rerank,
    embeddings_response_for,
    embeddings_texts,
    model_probe_path,
    passages_from_rerank,
    rerank_request_for,
    rerank_response_for,
    vectors_from_embeddings,
)


def passages(count: int) -> list[ScoredPassage]:
    return [
        ScoredPassage(id=f"p{i}", text=f"text {i}", retrieval_score=1.0 - i / 10)
        for i in range(count)
    ]


# --- paths -----------------------------------------------------------------


def test_paths_are_the_ovms_v3_routes() -> None:
    assert EMBEDDINGS_PATH == "/v3/embeddings"
    assert RERANK_PATH == "/v3/rerank"


@pytest.mark.parametrize(
    ("alias", "path"),
    [
        ("bge-reranker-base", "/v3/models/bge-reranker-base"),
        ("bge-small-en-v1.5", "/v3/models/bge-small-en-v1.5"),
    ],
)
def test_model_probe_path_names_the_alias(alias: str, path: str) -> None:
    assert model_probe_path(alias) == path


def test_unknown_model_error_is_the_ovms_text() -> None:
    expected = "Mediapipe graph definition with requested name is not found"
    assert UNKNOWN_MODEL_ERROR == expected


# --- request shapes --------------------------------------------------------


@pytest.mark.parametrize(
    ("request_body", "wire"),
    [
        (
            EmbeddingsRequest(model="e", input=["a", "b"]),
            {"model": "e", "input": ["a", "b"]},
        ),
        (EmbeddingsRequest(model="e", input="a"), {"model": "e", "input": "a"}),
        (
            RerankRequest(model="r", query="q", documents=["a"], top_n=1),
            {"model": "r", "query": "q", "documents": ["a"], "top_n": 1},
        ),
        (
            RerankRequest(model="r", query="q", documents=["a"]),
            {"model": "r", "query": "q", "documents": ["a"]},  # never "top_n": null
        ),
    ],
    ids=["embed-list", "embed-string", "rerank-top-n", "rerank-no-top-n"],
)
def test_requests_go_on_the_wire_without_nulls(
    request_body: EmbeddingsRequest | RerankRequest, wire: dict[str, object]
) -> None:
    assert request_body.to_wire() == wire


def test_requests_parse_from_plain_json() -> None:
    parsed = RerankRequest.model_validate(
        {"model": "r", "query": "q", "documents": ["a", "b"]}
    )
    assert (parsed.documents, parsed.top_n) == (["a", "b"], None)
    assert EmbeddingsRequest.model_validate({"model": "e", "input": "x"}).input == "x"


def test_rerank_request_for_sends_passage_texts_and_top_n() -> None:
    request = rerank_request_for("r", "q", passages(3), top_n=2)
    assert request.to_wire() == {
        "model": "r",
        "query": "q",
        "documents": ["text 0", "text 1", "text 2"],
        "top_n": 2,
    }


# --- responses, client side ------------------------------------------------


def test_embeddings_response_reads_ovms_json_and_ignores_extra_fields() -> None:
    body = {
        "object": "list",
        "model": "e",
        "usage": {"prompt_tokens": 2},
        "data": [{"object": "embedding", "index": 0, "embedding": [1, 2]}],
    }
    response = EmbeddingsResponse.model_validate(body)
    assert response.data[0].index == 0
    assert response.data[0].embedding == [1.0, 2.0]


@pytest.mark.parametrize(
    ("rows", "vectors"),
    [
        ([], []),
        ([(0, [0.1]), (1, [0.2])], [[0.1], [0.2]]),
        ([(2, [0.3]), (0, [0.1]), (1, [0.2])], [[0.1], [0.2], [0.3]]),
    ],
    ids=["empty", "in-order", "shuffled"],
)
def test_vectors_from_embeddings_follow_input_order(
    rows: list[tuple[int, list[float]]], vectors: list[list[float]]
) -> None:
    data = [{"index": i, "embedding": v} for i, v in rows]
    response = EmbeddingsResponse.model_validate({"data": data})
    assert vectors_from_embeddings(response) == vectors


@pytest.mark.parametrize(
    "body",
    [
        {"object": "list"},
        {"data": [{"embedding": [0.1]}]},
        {"data": [{"index": 0, "embedding": "not a vector"}]},
        {"results": [{"index": 0, "relevance_score": 0.1}]},
    ],
    ids=["no-data", "no-index", "bad-embedding", "rerank-body"],
)
def test_embeddings_response_rejects_bodies_off_the_shape(
    body: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        EmbeddingsResponse.model_validate(body)


def rerank_response(*results: tuple[int, float]) -> RerankResponse:
    rows = [{"index": i, "relevance_score": s} for i, s in results]
    return RerankResponse.model_validate({"results": rows})


@pytest.mark.parametrize(
    ("results", "top_n", "ids", "scores"),
    [
        (((0, 0.5), (2, 0.9)), 2, ["p2", "p0"], [0.9, 0.5]),
        (((0, 0.5), (1, 0.7), (2, 0.9)), 1, ["p2"], [0.9]),
        (((1, 0.2), (0, 0.4)), 5, ["p0", "p1"], [0.4, 0.2]),
        ((), 3, [], []),
    ],
    ids=["subset", "cut-to-top-n", "fewer-than-top-n", "no-results"],
)
def test_passages_from_rerank_map_back_by_index(
    results: tuple[tuple[int, float], ...],
    top_n: int,
    ids: list[str],
    scores: list[float],
) -> None:
    ranked = passages_from_rerank(rerank_response(*results), passages(3), top_n)
    assert [p.id for p in ranked] == ids
    assert [p.rerank_score for p in ranked] == scores


def test_passages_from_rerank_keep_the_retrieval_score() -> None:
    [ranked] = passages_from_rerank(rerank_response((1, 0.3)), passages(2), 1)
    assert (ranked.id, ranked.retrieval_score, ranked.text) == ("p1", 0.9, "text 1")


@pytest.mark.parametrize("index", [2, 7, -1])
def test_passages_from_rerank_reject_an_index_out_of_range(index: int) -> None:
    with pytest.raises(IndexError, match=f"index {index} .*2 documents"):
        passages_from_rerank(rerank_response((index, 0.9)), passages(2), 1)


@pytest.mark.parametrize(
    "body",
    [
        {"data": []},
        {"results": [{"index": 0}]},
        {"results": [{"relevance_score": 0.1}]},
        {"results": [{"index": "zero", "relevance_score": 0.1}]},
    ],
    ids=["no-results", "no-score", "no-index", "bad-index"],
)
def test_rerank_response_rejects_bodies_off_the_shape(body: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RerankResponse.model_validate(body)


# --- index bookkeeping, server side ----------------------------------------


@pytest.mark.parametrize(
    ("given", "texts"),
    [("one", ["one"]), (["a", "b"], ["a", "b"]), ([], [])],
    ids=["string", "list", "empty"],
)
def test_embeddings_texts_always_a_list(
    given: str | list[str], texts: list[str]
) -> None:
    assert embeddings_texts(EmbeddingsRequest(model="e", input=given)) == texts


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
    assert vectors_from_embeddings(response) == [[0.1, 0.2], [0.3, 0.4]]


def test_candidates_from_rerank_carry_their_document_index_as_id() -> None:
    request = RerankRequest(model="r", query="q", documents=["a", "b"])
    candidates = candidates_from_rerank(request)
    assert [(p.id, p.text) for p in candidates] == [("0", "a"), ("1", "b")]
    assert all(p.rerank_score is None for p in candidates)


def test_rerank_response_for_names_documents_by_index() -> None:
    request = RerankRequest(model="r", query="q", documents=["a", "b", "c"])
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


def test_rerank_round_trip_maps_back_to_the_client_passages() -> None:
    sent = passages(3)
    request = rerank_request_for("r", "q", sent, top_n=2)
    candidates = candidates_from_rerank(request)
    ranked = [candidates[1].model_copy(update={"rerank_score": 0.8})]
    [back] = passages_from_rerank(rerank_response_for(ranked), sent, 2)
    assert (back.id, back.rerank_score) == ("p1", 0.8)


def test_rerank_response_for_rejects_an_unscored_passage() -> None:
    [unscored] = candidates_from_rerank(
        RerankRequest(model="r", query="q", documents=["a"])
    )
    with pytest.raises(ValueError, match="'0' has no rerank_score"):
        rerank_response_for([unscored])
