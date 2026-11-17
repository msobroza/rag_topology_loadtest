"""The standard formats' request and answer shapes: validators and ``to_wire``."""

from __future__ import annotations

from query_processing.domain.standard_formats import (
    CohereRerankRequest,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIEmbeddingsRequest,
)

# --- OpenAI embeddings -----------------------------------------------------


def test_embeddings_request_reads_one_bare_string_as_one_text() -> None:
    # What our model server receives when a client sends ``"input": "hi"``.
    request = OpenAIEmbeddingsRequest.model_validate({"model": "m", "input": "hi"})
    assert request.to_wire() == {"model": "m", "input": ["hi"]}


# --- Cohere rerank ---------------------------------------------------------


def test_to_wire_leaves_unset_optional_fields_out() -> None:
    request = CohereRerankRequest(model="r", query="q", documents=[])
    assert request.to_wire() == {"model": "r", "query": "q", "documents": []}


# --- both, as the model server reads them ----------------------------------


def test_requests_parse_from_plain_json() -> None:
    parsed = CohereRerankRequest.model_validate(
        {"model": "r", "query": "q", "documents": ["a", "b"]}
    )
    assert (parsed.documents, parsed.top_n) == (["a", "b"], None)
    embeddings = OpenAIEmbeddingsRequest.model_validate({"model": "e", "input": "x"})
    assert embeddings.input == ["x"]


# --- OpenAI chat -------------------------------------------------------------


def test_a_chat_request_without_a_cap_is_the_request_of_before() -> None:
    request = OpenAIChatRequest(model="m", messages=[], temperature=0.0)
    assert request.to_wire() == {"model": "m", "messages": [], "temperature": 0.0}


def test_a_chat_request_carries_its_cap() -> None:
    request = OpenAIChatRequest(model="m", messages=[], temperature=0.0, max_tokens=128)
    assert request.to_wire()["max_tokens"] == 128


def test_a_chat_response_reads_its_usage_and_does_without_one() -> None:
    choices = [{"message": {"content": "Paris."}}]
    usage = {"prompt_tokens": 829, "completion_tokens": 128, "total_tokens": 957}
    response = OpenAIChatResponse.model_validate({"choices": choices, "usage": usage})
    assert response.usage is not None
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (
        829,
        128,
    )
    assert OpenAIChatResponse.model_validate({"choices": choices}).usage is None
