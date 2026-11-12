"""The registry: the only place that maps an API format name to its class.

A new format is one new module, one entry in the matching dict below, and one
new value in the matching ``Literal`` of ``domain/api_endpoint.py``. That holds for
an API that does the same operation with other field names: one request in, all
the vectors or scores out. An API that works differently (for example one call
per candidate) is not a format: it is an adapter class implementing the port.

Example:
    >>> embedding_api_format("openai")
    <OpenAIEmbeddingsFormat object>
    >>> rerank_api_format("jina")
    ValueError: unknown rerank API format 'jina' (expected one of: 'cohere')
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from .api_format_bases import (
    BaseEmbeddingApiFormat,
    BaseGenerationApiFormat,
    BaseRerankApiFormat,
)
from .cohere_rerank_format import CohereRerankFormat
from .openai_chat_format import OpenAIChatFormat
from .openai_embeddings_format import OpenAIEmbeddingsFormat

EMBEDDING_API_FORMATS: dict[str, type[BaseEmbeddingApiFormat]] = {
    "openai": OpenAIEmbeddingsFormat,
}
RERANK_API_FORMATS: dict[str, type[BaseRerankApiFormat]] = {
    "cohere": CohereRerankFormat,
}
GENERATION_API_FORMATS: dict[str, type[BaseGenerationApiFormat]] = {
    "openai": OpenAIChatFormat,
}

_Format = TypeVar("_Format")


def embedding_api_format(name: str) -> BaseEmbeddingApiFormat:
    """Return the embeddings API format called ``name``.

    Args:
        name: A key of ``EMBEDDING_API_FORMATS``, such as ``"openai"``.

    Returns:
        A fresh instance of that format.

    Example:
        >>> embedding_api_format("openai").request_body("m", ["a"])
        {'model': 'm', 'input': ['a']}
    """
    return _lookup("embedding", EMBEDDING_API_FORMATS, name)


def rerank_api_format(name: str) -> BaseRerankApiFormat:
    """Return the rerank API format called ``name``.

    Args:
        name: A key of ``RERANK_API_FORMATS``, such as ``"cohere"``.

    Returns:
        A fresh instance of that format.

    Example:
        >>> rerank_api_format("cohere").scores({"results": []})
        {}
    """
    return _lookup("rerank", RERANK_API_FORMATS, name)


def generation_api_format(name: str) -> BaseGenerationApiFormat:
    """Return the generation (chat) API format called ``name``.

    Args:
        name: A key of ``GENERATION_API_FORMATS``, such as ``"openai"``.

    Returns:
        A fresh instance of that format.

    Example:
        >>> generation_api_format("openai").answer({"choices": [{"message": {}}]})
        ''
    """
    return _lookup("generation", GENERATION_API_FORMATS, name)


def _lookup(kind: str, registry: Mapping[str, type[_Format]], name: str) -> _Format:
    # One error shape for the three registries: the bad name and the known ones.
    api_format = registry.get(name)
    if api_format is None:
        known = ", ".join(repr(known_name) for known_name in sorted(registry))
        raise ValueError(
            f"unknown {kind} API format {name!r} (expected one of: {known})"
        )
    return api_format()
