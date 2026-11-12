"""The registries that name each component's API formats, and the lookup."""

from __future__ import annotations

from collections.abc import Callable
from typing import get_args

import pytest

from rag_load_test.adapters.embedder.api.embedding_api_formats import (
    EMBEDDING_API_FORMATS,
    embedding_api_format,
)
from rag_load_test.adapters.embedder.api.openai_embeddings_format import (
    OpenAIEmbeddingsFormat,
)
from rag_load_test.adapters.llm.api.generation_api_formats import (
    GENERATION_API_FORMATS,
    generation_api_format,
)
from rag_load_test.adapters.llm.api.openai_chat_format import OpenAIChatFormat
from rag_load_test.adapters.reranker.api.cohere_rerank_format import CohereRerankFormat
from rag_load_test.adapters.reranker.api.rerank_api_formats import (
    RERANK_API_FORMATS,
    rerank_api_format,
)
from rag_load_test.domain.api_endpoint import (
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
