"""``RouterContract``: the tests a user runs on a router of theirs.

It is ``ComponentContract`` for the router: the three tests every component
passes, and two of the router's own, that the route chosen is one of the
router's routes and that its confidence is between 0 and 1. A user writes
``make_component``, ``make_request`` and ``routes``, the routes the router
chooses between, in the order its settings list them.

Example:
    >>> class TestNearestRouter(RouterContract):
    ...     def make_component(self) -> BaseRouter:
    ...         return NearestRouter.load("router.joblib")
    ...     def make_request(self) -> RouteRequest:
    ...         return RouteRequest(embedding=[0.1, 0.2])
    ...     def routes(self) -> Sequence[str]:
    ...         return ("self_contained", "leans_on_history")
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from ..adapters.router.base_router import BaseRouter
from ..domain.routes import RouteChoice, RouteRequest
from .component_contract import ComponentContract

# The router's settings name the embedder whose vectors it reads, and the
# settings refuse a name that is no embedder component. The contract serves
# the router alone: the embedder is declared, on its fake backend with a
# made-up source id, and never opened.
_EMBEDDER = "embedder"
_EMBEDDER_ENTRY = {"type": "embedder", "backend": "fake", "source_id": "contract/fake"}


class RouterContract(ComponentContract):
    """The tests every router passes, beyond those of every component.

    Example:
        >>> class TestFakeRouter(RouterContract):
        ...     def make_component(self) -> BaseRouter:
        ...         return FakeRouter(route="a")
        ...     def make_request(self) -> RouteRequest:
        ...         return RouteRequest(embedding=[0.1, 0.2])
        ...     def routes(self) -> Sequence[str]:
        ...         return ("a", "b")
    """

    @abstractmethod
    def make_component(self) -> BaseRouter:
        """Return a new router of the implementation under test.

        Returns:
            The router, ready to answer.

        Example:
            >>> TestFakeRouter().make_component()
            <FakeRouter ...>
        """

    @abstractmethod
    def make_request(self) -> RouteRequest:
        """Return the request every test sends: a query's vector.

        Returns:
            The request, a vector of the size the router reads.

        Example:
            >>> TestFakeRouter().make_request()
            RouteRequest(embedding=[0.1, 0.2])
        """

    @abstractmethod
    def routes(self) -> Sequence[str]:
        """Return the routes the router chooses between, as its settings list them.

        Returns:
            Their names, two at least.

        Example:
            >>> TestFakeRouter().routes()
            ('a', 'b')
        """

    def component_keys(self) -> dict[str, object]:
        """Return the router's own keys: an embedder's name and its routes.

        Returns:
            ``embedder`` and ``routes``, as a router's settings state them.

        Example:
            >>> TestFakeRouter().component_keys()
            {'embedder': 'embedder', 'routes': ['a', 'b']}
        """
        return {"embedder": _EMBEDDER, "routes": list(self.routes())}

    def workflow_keys(self) -> dict[str, object]:
        """Return what the router's settings name: its routes, and its embedder.

        Returns:
            ``routes``: a catalogue as ``routes.yaml`` would give it, each
            route with its name as description and utterance; ``components``:
            the embedder the router names, on its fake backend, never opened.

        Example:
            >>> TestFakeRouter().workflow_keys()["routes"][0]["name"]
            'a'
        """
        catalogue = [
            {"name": name, "description": name, "utterances": [name]}
            for name in self.routes()
        ]
        return {"routes": catalogue, "components": {_EMBEDDER: dict(_EMBEDDER_ENTRY)}}

    async def test_the_route_chosen_is_one_of_its_routes(self) -> None:
        """The router chooses one of the routes it lists.

        Raises:
            AssertionError: If it chose another; the message lists its routes.

        Example:
            >>> await TestFakeRouter().test_the_route_chosen_is_one_of_its_routes()
        """
        router, choice = await self._choice()
        routes = list(self.routes())
        assert choice.name in routes, (
            f"{type(router).__name__} chose {choice.name!r}; "
            f"expected one of its routes: {routes}"
        )

    async def test_the_confidence_is_between_0_and_1(self) -> None:
        """The router's confidence in its choice is a probability.

        Raises:
            AssertionError: If it is below 0 or above 1; the message shows it.

        Example:
            >>> await TestFakeRouter().test_the_confidence_is_between_0_and_1()
        """
        router, choice = await self._choice()
        assert 0.0 <= choice.confidence <= 1.0, (
            f"{type(router).__name__} chose {choice.name!r} with a "
            f"confidence of {choice.confidence!r}; expected one between 0 and 1"
        )

    async def _choice(self) -> tuple[BaseRouter, RouteChoice]:
        router = self.make_component()
        choice = await router.ainvoke(self.make_request())
        assert isinstance(
            choice, RouteChoice
        ), f"the router answered {choice!r}; expected a RouteChoice"
        return router, choice
