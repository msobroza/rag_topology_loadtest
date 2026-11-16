"""A router's settings: the embedder it reads, the routes it chooses between.

The router is a component type the toolkit registers on its class line, so a
router is read as a component of a registered type is, its backend's block
with the class its implementation names (``RegisteredComponentSettings``).
Its type's own keys are fields here, since its port cannot import the config:
``embedder``, the embedder component whose vectors it reads, and ``routes``,
the names of the routes it chooses between, two at least, each one of the
catalogue of ``routes.yaml`` (spec 8). A route the catalogue lacks is refused,
and so is an ``embedder`` that names no embedder component of the file.

Example:
    >>> router = RouterSettings.model_validate({"type": "router", "backend": "fake",
    ...     "embedder": "embedder", "routes": ["self_contained", "leans_on_history"]})
    >>> router.routes
    ['self_contained', 'leans_on_history']
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from ..domain.routes import Route
from .registered_component_settings import RegisteredComponentSettings
from .settings_blocks import ComponentSettings


class RouterSettings(RegisteredComponentSettings):
    """A router: the embedder it reads, its routes, and one block per backend.

    Example:
        >>> settings.components["router"].embedder
        'embedder'
    """

    type: Literal["router"] = "router"
    embedder: str  # the name of an embedder component of the workflow file
    routes: list[str] = Field(min_length=2)  # names of routes of routes.yaml

    def type_block(self) -> BaseModel | None:
        """Return the router's own keys: these settings, which hold them as fields.

        Returns:
            The settings themselves, whose ``embedder`` and ``routes`` an
            implementation and a plain call read.

        Example:
            >>> settings.components["router"].type_block().embedder
            'embedder'
        """
        return self


def routes_of(
    router: RouterSettings, catalogue: Mapping[str, Route]
) -> dict[str, Route]:
    """Return the routes a router lists, by name, in its order, from the catalogue.

    Args:
        router: The router's settings.
        catalogue: Every route of ``routes.yaml``, by name; the settings refuse
            a router that lists a route it lacks (``check_router_routes``).

    Returns:
        Each route it lists.

    Example:
        >>> catalogue = {route.name: route for route in settings.routes}
        >>> list(routes_of(settings.components["router"], catalogue))
        ['self_contained', 'leans_on_history']
    """
    return {name: catalogue[name] for name in router.routes}


def check_router_embedders(components: Mapping[str, ComponentSettings]) -> None:
    """Refuse a router whose ``embedder`` names no embedder component.

    Args:
        components: The components of the workflow file, by name.

    Raises:
        ValueError: If a router's ``embedder`` names a component the file does
            not declare, or one of another type; the message names the router,
            the value and the embedders the file declares.

    Example:
        >>> check_router_embedders(settings.components)
    """
    embedders = [n for n, c in components.items() if c.type == "embedder"]
    known = ", ".join(map(repr, embedders)) or "none"
    for name, router in components.items():
        if isinstance(router, RouterSettings) and router.embedder not in embedders:
            raise ValueError(
                f"router {name!r} names the embedder {router.embedder!r}, which is "
                f"no embedder component of the workflow file (its embedders: {known})"
            )


def check_router_routes(
    components: Mapping[str, ComponentSettings], routes: Sequence[Route], where: str
) -> None:
    """Refuse a route a router lists that the catalogue lacks, or a route named twice.

    Args:
        components: The components of the workflow file, by name.
        routes: The catalogue, as ``routes.yaml`` gives it.
        where: The routes file, as a refusal names it.

    Raises:
        ValueError: If two routes of the catalogue share a name, or a router
            lists a route it lacks; the message names the router, the route and
            the routes the catalogue has.

    Example:
        >>> check_router_routes(settings.components, settings.routes, "routes.yaml")
    """
    names = [route.name for route in routes]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise ValueError(f"the {where} names the route {twice[0]!r} twice")
    routers = {n: c for n, c in components.items() if isinstance(c, RouterSettings)}
    for name, router in routers.items():
        _refuse_missing_routes(name, router, names, where)


def _refuse_missing_routes(
    name: str, router: RouterSettings, names: list[str], where: str
) -> None:
    missing = [route for route in router.routes if route not in names]
    if missing:
        known = ", ".join(map(repr, sorted(names))) or "none"
        raise ValueError(
            f"router {name!r} lists the route {missing[0]!r}, which the {where} "
            f"lacks (expected one of its routes: {known})"
        )
