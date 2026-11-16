"""The router's test class catches a route it does not list and a confidence out of [0, 1]."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from query_processing.adapters.router.base_router import BaseRouter
from query_processing.domain.routes import RouteChoice, RouteRequest
from query_processing.testing import RouterContract


class ChoosingAlways(BaseRouter):
    """Answers one route choice, whatever the query."""

    def __init__(self, choice: RouteChoice) -> None:
        self.choice = choice

    def invoke(self, request: RouteRequest) -> RouteChoice:
        return self.choice


class ContractOf(RouterContract):
    """The contract of a router that always answers ``choice``, listing a and b."""

    def __init__(self, choice: RouteChoice) -> None:
        self.choice = choice

    def make_component(self) -> BaseRouter:
        return ChoosingAlways(self.choice)

    def make_request(self) -> RouteRequest:
        return RouteRequest(embedding=[0.1])

    def routes(self) -> Sequence[str]:
        return ("a", "b")


def test_the_contract_catches_a_route_the_router_does_not_list() -> None:
    wrong = ContractOf(RouteChoice(name="c", confidence=0.5))

    with pytest.raises(AssertionError, match=r"chose 'c'; expected one of its routes"):
        asyncio.run(wrong.test_the_route_chosen_is_one_of_its_routes())


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_the_contract_catches_a_confidence_out_of_0_and_1(confidence: float) -> None:
    wrong = ContractOf(RouteChoice(name="a", confidence=confidence))

    with pytest.raises(AssertionError, match="expected one between 0 and 1"):
        asyncio.run(wrong.test_the_confidence_is_between_0_and_1())


def test_a_router_that_keeps_to_its_routes_passes_both() -> None:
    right = ContractOf(RouteChoice(name="b", confidence=0.0))

    asyncio.run(right.test_the_route_chosen_is_one_of_its_routes())
    asyncio.run(right.test_the_confidence_is_between_0_and_1())
