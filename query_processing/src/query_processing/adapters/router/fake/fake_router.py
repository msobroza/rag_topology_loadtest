"""The ``fake`` router: always the first of its routes, with a confidence of 1.

Example:
    >>> await FakeRouter(route="self_contained").ainvoke(RouteRequest(embedding=[0.1]))
    RouteChoice(name='self_contained', confidence=1.0)
"""

from __future__ import annotations

from pydantic import BaseModel

from ....domain.components.component_resources import ComponentResources
from ....domain.components.routes import RouteChoice, RouteRequest
from ..base_router import BaseRouter


class FakeRouter(BaseRouter, backend="fake"):
    """Chooses one route whatever the query, so a graph's path is known in advance.

    Args:
        route: The route it always chooses.
    """

    def __init__(self, route: str) -> None:
        self.route = route

    @classmethod
    def from_settings(
        cls, settings: BaseModel | None, resources: ComponentResources
    ) -> FakeRouter:
        """Build it on the first of the routes the router lists.

        Args:
            settings: None: the fake has no block.
            resources: Its routes, in the order the router lists them.

        Returns:
            A fake router that chooses that route.

        Raises:
            ValueError: If the router lists no route.

        Example:
            >>> FakeRouter.from_settings(None, resources).route
            'self_contained'
        """
        del settings  # the fake has no block
        first = next(iter(resources.routes), None)
        if first is None:
            raise ValueError(
                f"router {resources.name!r} lists no route; expected the routes "
                "it chooses between, two at least, each in routes.yaml"
            )
        return cls(first)

    def invoke(self, request: RouteRequest) -> RouteChoice:
        """Choose its one route.

        Args:
            request: The query's vector, not read.

        Returns:
            Its route, with a confidence of 1.0.

        Example:
            >>> FakeRouter(route="a").invoke(RouteRequest(embedding=[0.1]))
            RouteChoice(name='a', confidence=1.0)
        """
        del request  # the fake chooses the same route for every query
        return RouteChoice(name=self.route, confidence=1.0)
