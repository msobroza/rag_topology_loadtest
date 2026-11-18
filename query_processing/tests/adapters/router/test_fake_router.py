"""The ``fake`` router passes the tests the toolkit ships for routers."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from query_processing.adapters.router.base_router import BaseRouter
from query_processing.adapters.router.fake.fake_router import FakeRouter
from query_processing.domain.component_resources import ComponentResources
from query_processing.domain.routes import Route, RouteChoice, RouteRequest
from query_processing.testing import RouterContract


class TestFakeRouter(RouterContract):
    def make_component(self) -> BaseRouter:
        return FakeRouter(route="a")

    def make_request(self) -> RouteRequest:
        return RouteRequest(embedding=[0.1, 0.2])

    def routes(self) -> Sequence[str]:
        return ("a", "b")


def router_resources(*routes: str) -> ComponentResources:
    """The resources of a router that lists ``routes``, in that order."""

    def no_other_component(other: str) -> object:
        raise ValueError(f"no component named {other!r} in this test")

    return ComponentResources(
        name="router",
        type_settings=None,
        component_settings=no_other_component,  # type: ignore[arg-type]
        routes={
            name: Route(name=name, description=name, utterances=[name])
            for name in routes
        },
    )


async def test_the_fake_router_built_from_settings_chooses_the_first_of_its_routes() -> (
    None
):
    router = FakeRouter.from_settings(None, router_resources("b", "a"))

    choice = await router.ainvoke(RouteRequest(embedding=[0.3]))

    assert choice == RouteChoice(name="b", confidence=1.0)


def test_a_fake_router_with_no_routes_is_refused() -> None:
    with pytest.raises(ValueError, match="router 'router' lists no route"):
        FakeRouter.from_settings(None, router_resources())
