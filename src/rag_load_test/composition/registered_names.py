"""An unknown backend or API format, refused before a command opens anything.

The openers and the API formats register themselves, in packages the settings
may not import, so the settings cannot refuse an unknown name. A component's
opener would refuse it, but only when that component opens, which can be after
another model has loaded. So each command that opens components calls
``check_registered_names`` while it reads its settings, and reports a refusal
as a usage error, as it reports any other settings mistake.

Example:
    >>> check_registered_names(RagSettings())  # the defaults: nothing raised
"""

from __future__ import annotations

from typing import Any

from ..adapters.embedder.api.base_embedding_api_format import EMBEDDING_API_FORMATS
from ..adapters.llm.api.base_generation_api_format import GENERATION_API_FORMATS
from ..adapters.reranker.api.base_rerank_api_format import RERANK_API_FORMATS
from ..config.component_settings import component_settings_of
from ..config.rag_settings import RagSettings
from ..domain.registry import Registry
from .openers.component_openers import (
    EMBEDDER_OPENERS,
    LLM_OPENERS,
    RERANKER_OPENERS,
    VECTOR_STORE_OPENERS,
)


def check_registered_names(settings: RagSettings) -> None:
    """Refuse a backend that has no opener, or an API format nobody registered.

    Every component is checked, whether the command opens it or not, as the
    settings checked every name before the API formats moved to a registry.

    Args:
        settings: The settings a command is about to open components from.

    Raises:
        ValueError: The first unknown name, with its sort and the known names:
            ``unknown generation API format 'jina' (expected one of: 'openai')``.

    Example:
        >>> check_registered_names(RagSettings(generation_api_format="jina"))
        ValueError: unknown generation API format 'jina' (expected one of: 'openai')
    """
    components = component_settings_of(settings)
    named: list[tuple[Registry[Any], str]] = [
        (EMBEDDER_OPENERS, components.embedder.backend),
        (RERANKER_OPENERS, components.reranker.backend),
        (LLM_OPENERS, components.llm.backend),
        (VECTOR_STORE_OPENERS, components.vector_store.backend),
        (EMBEDDING_API_FORMATS, components.embedder.api.api_format),
        (RERANK_API_FORMATS, components.reranker.api.api_format),
        (GENERATION_API_FORMATS, components.llm.api.api_format),
    ]
    for registry, name in named:
        registry.get(name)  # raises, naming the known names, when name is unknown
