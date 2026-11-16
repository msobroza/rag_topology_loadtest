"""The component types, each the port a component is used through.

A stage type declares the port of the component it calls; a component's type is
named after that port, as the glossary names it: ``embedder``, ``reranker``,
``llm``, ``vector_store`` and ``router``. Each port registers its type on its
class line in the domain's ``COMPONENT_TYPES``; this module imports the five
ports, so that reading the registry from here finds them registered. The
start-up checks need it before any component opens; which components a stage's
``call`` may name, and the type of each, come from the settings
(``WorkflowSettings.component_types``).

Example:
    >>> COMPONENT_TYPES.get(settings.component_types()["reranker"])
    <class 'BaseReranker'>
"""

from __future__ import annotations

from ..adapters.embedder.base_embedder import BaseEmbedder
from ..adapters.llm.base_llm import BaseLLM
from ..adapters.reranker.base_reranker import BaseReranker
from ..adapters.router.base_router import BaseRouter
from ..adapters.vector_store.base_vector_store import BaseVectorStore
from ..domain.base_component import COMPONENT_TYPES

# The ports are imported for their registrations, and named here so the imports
# are kept: a port registers its component type when its class is defined.
__all__ = [
    "COMPONENT_TYPES",
    "BaseEmbedder",
    "BaseLLM",
    "BaseReranker",
    "BaseRouter",
    "BaseVectorStore",
]
