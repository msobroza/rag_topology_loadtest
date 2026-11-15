"""The component types, each the port a component is used through, and the components.

A stage type declares the port of the component it calls; a component's type is
named after that port, as the glossary names it: ``embedder``, ``reranker``,
``llm`` and ``vector_store``. The start-up checks need both before any
component opens: which components a stage's ``call`` may name, and the type of
each.

Example:
    >>> COMPONENT_TYPES[declared_components()["reranker"]]
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


def declared_components() -> dict[str, str]:
    """Return the components a stage's ``call`` may name, each with its type.

    The one place that lists them before any opens. Today they are fixed: one
    component of each type, named after it, as ``RagDependencies.components``
    gives them once open. When the workflow file declares its components, their
    names and types come from it instead.

    Returns:
        Each component's type, by the component's name.

    Example:
        >>> declared_components()["vector_store"]
        'vector_store'
    """
    return {component_type: component_type for component_type in COMPONENT_TYPES}
