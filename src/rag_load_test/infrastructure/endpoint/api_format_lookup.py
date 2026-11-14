"""How an API format is found by name, for every component's registry.

Each component's ``api`` directory keeps its own registry of formats
(``embedding_api_formats.py``, ``rerank_api_formats.py``,
``generation_api_formats.py``). A format is for an API that does the same
operation with other field names: one request in, all the vectors or scores
out. An API that works differently (for example one call per candidate) is
not a format: it is an implementation of the port.

Example:
    >>> lookup_api_format("rerank", RERANK_API_FORMATS, "jina")
    ValueError: unknown rerank API format 'jina' (expected one of: 'cohere')
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

_Format = TypeVar("_Format")


def lookup_api_format(
    kind: str, registry: Mapping[str, type[_Format]], name: str
) -> _Format:
    """Return a fresh instance of the format ``name`` of ``registry``.

    Args:
        kind: The component's word in the error: ``embedding``, ``rerank``
            or ``generation``.
        registry: Format names to their classes.
        name: The format asked for.

    Returns:
        A fresh instance of that format.

    Raises:
        ValueError: If ``registry`` has no such name; the message names the
            known ones.

    Example:
        >>> lookup_api_format("embedding", EMBEDDING_API_FORMATS, "openai")
        <OpenAIEmbeddingsFormat object>
    """
    api_format = registry.get(name)
    if api_format is None:
        known = ", ".join(repr(known_name) for known_name in sorted(registry))
        raise ValueError(
            f"unknown {kind} API format {name!r} (expected one of: {known})"
        )
    return api_format()
