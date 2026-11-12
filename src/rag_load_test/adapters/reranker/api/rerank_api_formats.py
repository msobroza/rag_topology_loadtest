"""The rerank API formats by name: the one place that maps a name to its class.

A new format is one new module beside this one, one entry below, and one
new value in the matching ``Literal`` of ``domain/api_endpoint.py``.

Example:
    >>> rerank_api_format("cohere")
    <CohereRerankFormat object>
"""

from __future__ import annotations

from ...endpoint.api_format_lookup import lookup_api_format
from .base_rerank_api_format import BaseRerankApiFormat
from .cohere_rerank_format import CohereRerankFormat

RERANK_API_FORMATS: dict[str, type[BaseRerankApiFormat]] = {
    "cohere": CohereRerankFormat,
}


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
    return lookup_api_format("rerank", RERANK_API_FORMATS, name)
