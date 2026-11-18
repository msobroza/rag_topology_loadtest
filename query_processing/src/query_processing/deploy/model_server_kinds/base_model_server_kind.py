"""What a model-server kind is, and the registry of them by name.

A model-server kind is what serves a component in a model server App: our
FastAPI model server, or OpenVINO Model Server. Each kind is one class
registered here, in a module of its own beside this one, and one hand-written
launch script, ``domino/model_servers/<name>.sh``, that holds its bootstrap and
its command (shell, because the bootstrap runs before Python is installed). The
class says which component types the kind serves, on which route, and how a
served model is probed; a kind that serves the registered component types (an
extension's, ``registered_backends.py``) serves each on its call route
(``domain/endpoints/api_endpoint.py``). The deploy tool builds the URL it hands
the workflow App from that route, and refuses a component no kind can serve
before anything is published. A new kind is one module here and one script:
the deploy tool's flow and ``domino/app.sh`` do not change.

Example:
    >>> model_server_kind("ovms").route_of("reranker")
    '/v3/rerank'
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ...domain.endpoints.api_endpoint import call_path
from ...domain.registry import Registry
from ..registered_backends import is_registered_type

if TYPE_CHECKING:  # deploy_settings checks a kind's name against the registry
    from ..deploy_settings import DeploySettings


class BaseModelServerKind(ABC):
    """What serves a component in a model server App, as the deploy tool sees it.

    Example:
        >>> model_server_kind("fastapi").serves("llm")
        False
    """

    @property
    @abstractmethod
    def routes(self) -> Mapping[str, str]:
        """The route each component type it serves is called on, by type."""

    @property
    @abstractmethod
    def probe_route(self) -> str:
        """The route that says whether a model is served; ``{name}`` is its alias."""

    @property
    def serves_registered_types(self) -> bool:
        """Whether it serves every registered component type, on its call route."""
        return False

    def serves(self, component_type: str) -> bool:
        """Say whether this kind can serve a component of ``component_type``.

        Args:
            component_type: The component's type, such as ``reranker``.

        Returns:
            Whether the kind has a route for that type.

        Example:
            >>> model_server_kind("ovms").serves("embedder")
            True
        """
        return self._route_or_none(component_type) is not None

    def route_of(self, component_type: str) -> str:
        """Return the route a served component of ``component_type`` is called on.

        Args:
            component_type: The served component's type.

        Returns:
            The route, such as ``/v3/rerank``, to append to the App's URL; a
            registered component type's call route, such as ``/call/shouter``.

        Raises:
            ValueError: If the kind serves no component of that type; the
                message names the types it serves.

        Example:
            >>> model_server_kind("fastapi").route_of("embedder")
            '/v3/embeddings'
        """
        route = self._route_or_none(component_type)
        if route is not None:
            return route
        registered = " or a registered type" if self.serves_registered_types else ""
        raise ValueError(
            f"this model-server kind serves no component of type {component_type!r} "
            f"(expected one of: {', '.join(map(repr, self.routes))}{registered})"
        )

    def _route_or_none(self, component_type: str) -> str | None:
        if component_type in self.routes:
            return self.routes[component_type]
        if self.serves_registered_types and is_registered_type(component_type):
            return call_path(component_type)
        return None

    def launch_exports(self, settings: DeploySettings) -> dict[str, str]:
        """Give what its launch script reads beyond the component and the kind.

        Args:
            settings: The deploy settings.

        Returns:
            The variables to export in the App's entry script: none by default.

        Example:
            >>> model_server_kind("ovms").launch_exports(settings)
            {'QP_OVMS_BUNDLE': '/mnt/data/ovms.tar.gz'}
        """
        del settings  # a kind whose script needs nothing more ignores them
        return {}

    def refuse_unusable_settings(self, settings: DeploySettings) -> None:
        """Refuse, before anything is published, settings its Apps cannot start with.

        Args:
            settings: The deploy settings.

        Example:
            >>> model_server_kind("fastapi").refuse_unusable_settings(settings)
        """
        del settings  # a kind that needs no setting of its own accepts any


# Every model-server kind by name; a served component's App block names one.
MODEL_SERVER_KINDS: Registry[type[BaseModelServerKind]] = Registry("model-server kind")


def model_server_kind(name: str) -> BaseModelServerKind:
    """Return the model-server kind registered under ``name``.

    Args:
        name: The kind's name, as the deploy settings give it.

    Returns:
        The kind.

    Raises:
        ValueError: If no kind is registered under ``name``; the message names
            every registered kind.

    Example:
        >>> model_server_kind("fastapi").route_of("reranker")
        '/v3/rerank'
    """
    return MODEL_SERVER_KINDS.get(name)()
