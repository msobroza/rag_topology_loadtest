"""``ComponentFactory``: a component opened once, by the opener of its type and backend.

The factory knows no component type and no backend: the registry of a type's
openers (``openers/component_openers.py``) gives the opener its backend names,
and an unknown backend fails naming the known ones. A component is opened
under its name, so two components of one type are two resources. A port
handed in ready-made (a test double, mostly) under a component's name is used
as given and never closed here: the factory registers nothing on the scope
for it.

Example:
    >>> factory = ComponentFactory(shared, {"reranker": None})
    >>> reranker = await factory.open(RERANKER_OPENERS, "reranker", reranker_settings)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ..config.registered_component_settings import RegisteredComponentSettings
from ..domain.base_component import BaseComponent
from ..domain.component_resources import ComponentResources
from .openers.component_openers import (
    ComponentOpeners,
    P,
    S,
    SharedResources,
)
from .registered_components import open_registered

# The backend whose model runs in this process; a model server always opens it.
LOCAL = "local"


class ComponentFactory:
    """Opens each component once, by the opener registered for its type and backend.

    Args:
        shared: What every opener is given; its scope opens each component once
            and closes what the openers registered.
        supplied: Ready-made ports by component name (``embedder``,
            ``reranker``, ...); None builds one.

    Example:
        >>> factory = ComponentFactory(shared, {"llm": FakeLLM()})
        >>> await factory.open(LLM_OPENERS, "llm", llm_settings)  # the FakeLLM given
    """

    def __init__(self, shared: SharedResources, supplied: Mapping[str, object]) -> None:
        self._shared = shared
        self._supplied = {
            name: port for name, port in supplied.items() if port is not None
        }

    async def open(self, openers: ComponentOpeners[S, P], name: str, component: S) -> P:
        """Open the component: the supplied port, else the one its backend names.

        Args:
            openers: The openers of the component's type.
            name: The component's name, which it is opened and supplied under.
            component: The component's settings; ``backend`` picks the opener.

        Returns:
            The port, the same object on every call within the scope.

        Raises:
            RagDependencyError: If the backend has no opener, naming the known
                ones, or if the opener fails.

        Example:
            >>> reranker = await factory.open(RERANKER_OPENERS, "reranker", settings)
        """
        supplied = self._supplied.get(name)
        if supplied is not None:
            # The caller asks for the port of the component's type.
            return cast(P, supplied)
        backend = component.backend
        return await self._shared.scope.once(
            _resource_name(name, backend),
            backend,
            lambda: self._open_backend(openers, component),
        )

    async def open_registered(
        self,
        name: str,
        component: RegisteredComponentSettings,
        resources: ComponentResources,
    ) -> BaseComponent[Any, Any]:
        """Open a component of a type registered in ``COMPONENT_TYPES``, once.

        It has no opener: the implementation its backend names builds itself.

        Args:
            name: The component's name, which it is opened and supplied under.
            component: The component's settings, its backend's block read.
            resources: What its implementation may use besides its block.

        Returns:
            The supplied component, else the one its backend names, the same
            object on every call within the scope.

        Raises:
            RagDependencyError: If no implementation names the backend, naming
                those that do, or if building it fails.

        Example:
            >>> shouter = await factory.open_registered("shouter", settings, resources)
        """
        supplied = self._supplied.get(name)
        if supplied is not None:
            # A ready-made component handed in under this name, as for the others.
            return cast(BaseComponent[Any, Any], supplied)
        return await self._shared.scope.once(
            name,
            component.backend,
            lambda: open_registered(name, component, resources),
        )

    async def open_local(
        self, openers: ComponentOpeners[S, P], name: str, component: S
    ) -> P:
        """Open the component on the ``local`` backend, whatever its settings say.

        A model server is what the other backends call, so it always loads the
        in-process model.

        Args:
            openers: The openers of the component's type.
            name: The component's name.
            component: The component's settings; its backend is not read.

        Returns:
            The supplied port if one was given, else the in-process one.

        Example:
            >>> embedder = await factory.open_local(EMBEDDER_OPENERS, "embedder", e)
        """
        local = component.model_copy(update={"backend": LOCAL})
        return await self.open(openers, name, local)

    async def _open_backend(self, openers: ComponentOpeners[S, P], component: S) -> P:
        return await openers.get(component.backend)(component, self._shared)


def _resource_name(name: str, backend: str) -> str:
    # An in-process model is loaded once per process, whether the workflow app
    # opens it by its backend or a model server whatever its backend says; its
    # errors have always named it local_<name>.
    return f"local_{name}" if backend == LOCAL else name
