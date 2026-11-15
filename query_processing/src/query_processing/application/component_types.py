"""The component types, each the port a component is used through.

A stage type declares the port of the component it calls; a component's type is
named after that port, as the glossary names it: ``embedder``, ``reranker``,
``llm`` and ``vector_store``. The start-up checks need it before any component
opens; which components a stage's ``call`` may name, and the type of each, come
from the settings (``WorkflowSettings.component_types``).

Example:
    >>> COMPONENT_TYPES[settings.component_types()["reranker"]]
    <class 'BaseReranker'>
"""

from __future__ import annotations

from typing import Any

from ..adapters.embedder.base_embedder import BaseEmbedder
from ..adapters.llm.base_llm import BaseLLM
from ..adapters.reranker.base_reranker import BaseReranker
from ..adapters.vector_store.base_vector_store import BaseVectorStore

# A component type's name, and the port a component of that type is used through.
COMPONENT_TYPES: dict[str, type[Any]] = {
    "embedder": BaseEmbedder,
    "reranker": BaseReranker,
    "llm": BaseLLM,
    "vector_store": BaseVectorStore,
}
