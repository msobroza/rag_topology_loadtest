"""The standard formats' request and answer shapes: validators and ``to_wire``."""

from __future__ import annotations

from rag_load_test.domain.standard_formats import (
    CohereRerankRequest,
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
