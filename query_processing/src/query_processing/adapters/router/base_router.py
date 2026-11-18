"""The router port: chooses, for a query's vector, one route out of its own list.

The router is the toolkit's fifth component type, registered as ``router`` on
its class line. Its settings name the embedder whose vectors it reads and the
routes it chooses between, out of the catalogue of ``routes.yaml`` (spec 8).
Only how it chooses is a use case's: an implementation writes ``invoke`` or
``ainvoke``, and ``from_settings`` to read its block and its routes. The
toolkit gives the ``fake`` backend, and builds the ``served`` one. This module
imports the domain only.

Example:
    >>> class NearestRouter(BaseRouter, backend="local", settings=NearestSettings):
    ...     def invoke(self, request: RouteRequest) -> RouteChoice: ...
"""

from __future__ import annotations

from pydantic import BaseModel

from ...domain.components.base_component import BaseComponent
from ...domain.components.routes import RouteChoice, RouteRequest


class BaseRouter(BaseComponent[RouteRequest, RouteChoice], component_type="router"):
    """Chooses one of its routes for a query's vector, with its confidence."""

    @classmethod
    def answer_labels(
        cls, type_settings: BaseModel | None
    ) -> dict[str, tuple[str, ...]]:
        """Return the router's routes, the labels of the route chosen.

        A plain call on a router gives them to the field it writes the route's
        name to, and a label edge that reads it has one row for each.

        Args:
            type_settings: The router's settings, which list its ``routes``.

        Returns:
            ``{"name": its routes}``.

        Raises:
            TypeError: If the settings list no routes.

        Example:
            >>> BaseRouter.answer_labels(settings.components["router"])
            {'name': ('self_contained', 'leans_on_history')}
        """
        routes = getattr(type_settings, "routes", None)
        if not isinstance(routes, list | tuple):
            raise TypeError(
                f"a router's settings are {type_settings!r}; expected settings "
                "that list its routes, such as routes: [a, b]"
            )
        return {"name": tuple(str(route) for route in routes)}
