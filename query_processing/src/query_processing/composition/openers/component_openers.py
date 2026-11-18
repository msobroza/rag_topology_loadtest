"""What an opener is, what it is given, and the registry of openers of each type.

An opener opens one component on one backend: it takes the component's
settings, in the workflow file's shape
(``config/components/component_settings.py``), and the resources the process
shares, and returns the open port. Each component type has its registry of
openers, by backend. A new backend is one new module
beside this one, its opener decorated with ``@<TYPE>_OPENERS.register(name)``;
the package imports every module, and ``ComponentFactory`` finds the opener.

Example:
    >>> @LLM_OPENERS.register("fake")
    ... async def open_llm_on_fake(component, shared): return FakeLLM()
    >>> LLM_OPENERS.get("echo")
    ValueError: unknown llm backend 'echo' (expected one of: 'api', 'fake', 'served')
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ...adapters.embedder.base_embedder import BaseEmbedder
from ...adapters.llm.base_llm import BaseLLM
from ...adapters.reranker.base_reranker import BaseReranker
from ...adapters.vector_store.base_vector_store import BaseVectorStore
from ...config.components.component_settings import (
    ComponentSettings,
    EmbedderSettings,
    LlmSettings,
    RerankerSettings,
    VectorStoreSettings,
)
from ...domain.registry import Registry
from ..http_resources import HttpResources
from ..local_model_resources import LocalModelResources
from ..resource_scope import Opener, ResourceScope

S = TypeVar("S", bound=ComponentSettings)  # a component type's settings
P = TypeVar("P")  # the port it is used through


@dataclass(frozen=True)
class SharedResources:
    """What the process shares with every opener.

    Example:
        >>> shared = SharedResources(scope, http, local_models, open_embedder)
        >>> client = await shared.http.client()
    """

    scope: ResourceScope  # closes what an opener registers on its exit stack
    http: HttpResources  # the shared HTTP client, policy and run tokens
    local_models: LocalModelResources  # the budget, pool, files and calibration
    # The embedder that seeds a fake store: the one its searches use
    # (config/searched_embedder.py, seeding_embedder), else the only one declared.
    open_embedder: Opener[BaseEmbedder]


# Opens one component on one backend.
ComponentOpener = Callable[[S, SharedResources], Awaitable[P]]


class ComponentOpeners(Registry[ComponentOpener[S, P]], Generic[S, P]):
    """The openers of one component type, by backend.

    Args:
        component_type: The type, such as ``embedder``: named in every error,
            and what a port handed in ready-made is keyed by.

    Example:
        >>> EMBEDDER_OPENERS.get("local")
        <function open_embedder_on_local ...>
    """

    def __init__(self, component_type: str) -> None:
        super().__init__(f"{component_type} backend")
        self.component_type = component_type


EMBEDDER_OPENERS: ComponentOpeners[EmbedderSettings, BaseEmbedder] = ComponentOpeners(
    "embedder"
)
RERANKER_OPENERS: ComponentOpeners[RerankerSettings, BaseReranker] = ComponentOpeners(
    "reranker"
)
LLM_OPENERS: ComponentOpeners[LlmSettings, BaseLLM] = ComponentOpeners("llm")
VECTOR_STORE_OPENERS: ComponentOpeners[VectorStoreSettings, BaseVectorStore] = (
    ComponentOpeners("vector_store")
)
# The openers of each component type, by the type a component's settings name.
OPENERS_BY_TYPE: dict[str, ComponentOpeners[Any, Any]] = {
    openers.component_type: openers
    for openers in (
        EMBEDDER_OPENERS,
        RERANKER_OPENERS,
        LLM_OPENERS,
        VECTOR_STORE_OPENERS,
    )
}
