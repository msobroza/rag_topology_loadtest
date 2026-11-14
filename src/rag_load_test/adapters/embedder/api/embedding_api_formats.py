"""The embeddings API formats by name: the one place that maps a name to its class.

A new format is one new module beside this one, one entry below, and one
new value in the matching ``Literal`` of ``domain/api_endpoint.py``.

Example:
    >>> embedding_api_format("openai")
    <OpenAIEmbeddingsFormat object>
"""

from __future__ import annotations

from ....infrastructure.endpoint.api_format_lookup import lookup_api_format
from .base_embedding_api_format import BaseEmbeddingApiFormat
from .openai_embeddings_format import OpenAIEmbeddingsFormat

EMBEDDING_API_FORMATS: dict[str, type[BaseEmbeddingApiFormat]] = {
    "openai": OpenAIEmbeddingsFormat,
}


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
    return lookup_api_format("embedding", EMBEDDING_API_FORMATS, name)
