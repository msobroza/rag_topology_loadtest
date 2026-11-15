"""Each component's API formats are found by name through its registry."""

from __future__ import annotations

import pytest

from rag_load_test.adapters.embedder.api.base_embedding_api_format import (
    EMBEDDING_API_FORMATS,
)
from rag_load_test.adapters.embedder.api.openai_embeddings_format import (
    OpenAIEmbeddingsFormat,
)
from rag_load_test.adapters.llm.api.base_generation_api_format import (
    GENERATION_API_FORMATS,
)
from rag_load_test.adapters.llm.api.openai_chat_format import OpenAIChatFormat
from rag_load_test.adapters.reranker.api.base_rerank_api_format import (
    RERANK_API_FORMATS,
)
from rag_load_test.adapters.reranker.api.cohere_rerank_format import CohereRerankFormat
from rag_load_test.domain.registry import Registry
from rag_load_test.domain.standard_formats import STANDARD_API_FORMATS

REGISTRIES: dict[str, Registry[type]] = {
    "embedding": EMBEDDING_API_FORMATS,
    "rerank": RERANK_API_FORMATS,
    "generation": GENERATION_API_FORMATS,
}


@pytest.mark.parametrize(
    ("component", "name", "expected"),
    [
        ("embedding", "openai", OpenAIEmbeddingsFormat),
        ("rerank", "cohere", CohereRerankFormat),
        ("generation", "openai", OpenAIChatFormat),
    ],
)
def test_each_shipped_format_is_registered_under_its_name(
    component: str, name: str, expected: type
) -> None:
    assert REGISTRIES[component].get(name) is expected


@pytest.mark.parametrize(
    ("component", "known"),
    [("embedding", "'openai'"), ("rerank", "'cohere'"), ("generation", "'openai'")],
)
def test_an_unknown_format_names_the_component_and_the_known_formats(
    component: str, known: str
) -> None:
    with pytest.raises(ValueError, match="unknown") as refused:
        REGISTRIES[component].get("jina")

    assert str(refused.value) == (
        f"unknown {component} API format 'jina' (expected one of: {known})"
    )


@pytest.mark.parametrize(
    ("component_type", "component"),
    [("embedder", "embedding"), ("reranker", "rerank"), ("llm", "generation")],
)
def test_the_standard_format_of_each_component_type_is_registered(
    component_type: str, component: str
) -> None:
    # A served component is called in it, so it must be one the registry knows.
    name = STANDARD_API_FORMATS[component_type]

    assert REGISTRIES[component].get(name) is not None
