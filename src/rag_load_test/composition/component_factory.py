"""``ComponentFactory``: a component opened once, by the opener of its type and backend.

The factory knows no component type and no backend: the registry of a type's
openers (``openers/component_openers.py``) gives the opener its backend names,
and an unknown backend fails naming the known ones. A port handed in ready-made
(a test double, mostly) is used as given and never closed here: the factory
registers nothing on the scope for it.

Example:
    >>> factory = ComponentFactory(shared, {"reranker": None})
    >>> reranker = await factory.open(RERANKER_OPENERS, components.reranker)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from .openers.component_openers import (
    ComponentOpeners,
    P,
    S,
    SharedResources,
)

# The backend whose model runs in this process; a model server always opens it.
LOCAL = "local"


class ComponentFactory:
    """Opens each component once, by the opener registered for its type and backend.

    Args:
        shared: What every opener is given; its scope opens each component once
            and closes what the openers registered.
        supplied: Ready-made ports by component type (``embedder``,
            ``reranker``, ``llm``, ``vector_store``); None builds one.

    Example:
        >>> factory = ComponentFactory(shared, {"llm": FakeLLM()})
        >>> await factory.open(LLM_OPENERS, components.llm)  # the FakeLLM given
    """

    def __init__(self, shared: SharedResources, supplied: Mapping[str, object]) -> None:
        self._shared = shared
        self._supplied = {
            component_type: port
            for component_type, port in supplied.items()
            if port is not None
        }

    async def open(self, openers: ComponentOpeners[S, P], component: S) -> P:
        """Open the component: the supplied port, else the one its backend names.

        Args:
            openers: The openers of the component's type.
            component: The component's settings; ``backend`` picks the opener.

        Returns:
            The port, the same object on every call within the scope.

        Raises:
            RagDependencyError: If the backend has no opener, naming the known
                ones, or if the opener fails.

        Example:
            >>> reranker = await factory.open(RERANKER_OPENERS, components.reranker)
        """
        supplied = self._supplied.get(openers.component_type)
        if supplied is not None:
            # CompositionRoot's SuppliedPorts types each port by its component type.
            return cast(P, supplied)
        backend = component.backend
        return await self._shared.scope.once(
            _resource_name(openers.component_type, backend),
            backend,
            lambda: self._open_backend(openers, component),
        )

    async def open_local(self, openers: ComponentOpeners[S, P], component: S) -> P:
        """Open the component on the ``local`` backend, whatever its settings say.

        A model server is what the other backends call, so it always loads the
        in-process model.

        Args:
            openers: The openers of the component's type.
            component: The component's settings; its backend is not read.

        Returns:
            The supplied port if one was given, else the in-process one.

        Example:
            >>> embedder = await factory.open_local(EMBEDDER_OPENERS, embedder)
        """
        return await self.open(openers, component.model_copy(update={"backend": LOCAL}))

    async def _open_backend(self, openers: ComponentOpeners[S, P], component: S) -> P:
        return await openers.get(component.backend)(component, self._shared)


def _resource_name(component_type: str, backend: str) -> str:
    # An in-process model is loaded once per process, whether the workflow app
    # opens it by its backend or a model server whatever its backend says; its
    # errors have always named it local_<type>.
    return f"local_{component_type}" if backend == LOCAL else component_type
