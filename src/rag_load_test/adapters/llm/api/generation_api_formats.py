"""The generation API formats by name: the one place that maps a name to its class.

A new format is one new module beside this one, one entry below, and one
new value in the matching ``Literal`` of ``domain/api_endpoint.py``.

Example:
    >>> generation_api_format("openai")
    <OpenAIChatFormat object>
"""

from __future__ import annotations

from ...endpoint.api_format_lookup import lookup_api_format
from .base_generation_api_format import BaseGenerationApiFormat
from .openai_chat_format import OpenAIChatFormat

GENERATION_API_FORMATS: dict[str, type[BaseGenerationApiFormat]] = {
    "openai": OpenAIChatFormat,
}


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
    return lookup_api_format("generation", GENERATION_API_FORMATS, name)
