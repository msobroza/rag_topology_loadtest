"""The API formats (pure data transformations) and the registry that names them."""

from __future__ import annotations

from collections.abc import Callable
from typing import get_args

import pytest
from pydantic import ValidationError

from rag_load_test.adapters.model_api.api_formats import (
    EMBEDDING_API_FORMATS,
    GENERATION_API_FORMATS,
    RERANK_API_FORMATS,
    embedding_api_format,
    generation_api_format,
    rerank_api_format,
)
from rag_load_test.adapters.model_api.cohere_rerank_format import (
    CohereRerankFormat,
    CohereRerankRequest,
)
from rag_load_test.adapters.model_api.openai_chat_format import OpenAIChatFormat
from rag_load_test.adapters.model_api.openai_embeddings_format import (
    OpenAIEmbeddingsFormat,
    OpenAIEmbeddingsRequest,
)
from rag_load_test.domain.model_api import (
    EmbeddingApiFormat,
    GenerationApiFormat,
    RerankApiFormat,
)

# --- the registry ----------------------------------------------------------


@pytest.mark.parametrize(
    ("registry", "names"),
    [
        (EMBEDDING_API_FORMATS, EmbeddingApiFormat),
        (RERANK_API_FORMATS, RerankApiFormat),
        (GENERATION_API_FORMATS, GenerationApiFormat),
    ],
    ids=["embedding", "rerank", "generation"],
)
def test_the_registry_knows_exactly_the_settings_names(
    registry: dict[str, type], names: object
) -> None:
    assert set(registry) == set(get_args(names))


@pytest.mark.parametrize(
    ("lookup", "name", "expected"),
    [
        (embedding_api_format, "openai", OpenAIEmbeddingsFormat),
        (rerank_api_format, "cohere", CohereRerankFormat),
        (generation_api_format, "openai", OpenAIChatFormat),
    ],
)
def test_lookups_return_an_instance(
    lookup: Callable[[str], object], name: str, expected: type
) -> None:
    assert isinstance(lookup(name), expected)


@pytest.mark.parametrize(
    ("lookup", "kind", "known"),
    [
        (embedding_api_format, "embedding", "'openai'"),
        (rerank_api_format, "rerank", "'cohere'"),
        (generation_api_format, "generation", "'openai'"),
    ],
)
def test_an_unknown_name_raises_with_the_known_names(
    lookup: Callable[[str], object], kind: str, known: str
) -> None:
    with pytest.raises(ValueError, match="unknown") as info:
        lookup("jina")
    assert str(info.value) == (
        f"unknown {kind} API format 'jina' (expected one of: {known})"
    )


# --- OpenAI embeddings -----------------------------------------------------


def test_embeddings_request_sends_input_as_a_list() -> None:
    body = OpenAIEmbeddingsFormat().request_body("m", ("only",))
    assert body == {"model": "m", "input": ["only"]}


def test_embeddings_request_reads_one_bare_string_as_one_text() -> None:
    # What our model server receives when a client sends ``"input": "hi"``.
    request = OpenAIEmbeddingsRequest.model_validate({"model": "m", "input": "hi"})
    assert request.to_wire() == {"model": "m", "input": ["hi"]}


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


# --- Cohere rerank ---------------------------------------------------------


def test_rerank_request_has_the_cohere_shape() -> None:
    body = CohereRerankFormat().request_body("r", "q", ("a", "b"), 1)
    assert body == {"model": "r", "query": "q", "documents": ["a", "b"], "top_n": 1}


def test_to_wire_leaves_unset_optional_fields_out() -> None:
    request = CohereRerankRequest(model="r", query="q", documents=[])
    assert request.to_wire() == {"model": "r", "query": "q", "documents": []}


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


# --- OpenAI chat -----------------------------------------------------------


def test_chat_request_has_the_openai_shape() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    body = OpenAIChatFormat().request_body("gpt", messages, 0.3)
    assert body == {"model": "gpt", "messages": messages, "temperature": 0.3}


@pytest.mark.parametrize(("content", "expected"), [("Paris.", "Paris."), (None, "")])
def test_chat_answer_reads_the_first_choice(content: str | None, expected: str) -> None:
    body = {"choices": [{"message": {"content": content}}, {"message": {}}]}
    assert OpenAIChatFormat().answer(body) == expected


@pytest.mark.parametrize(
    "body", [{"choices": []}, {}, {"choices": [{}]}], ids=["empty", "none", "no-msg"]
)
def test_chat_malformed_bodies_fail_validation(body: object) -> None:
    with pytest.raises(ValidationError):
        OpenAIChatFormat().answer(body)
