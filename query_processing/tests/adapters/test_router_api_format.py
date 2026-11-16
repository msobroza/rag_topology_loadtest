"""The router's API format, written by hand: its answer on the wire names ``route``."""

from __future__ import annotations

import pytest

from query_processing.adapters.router.base_router import BaseRouter
from query_processing.adapters.router.router_api_format import RouterApiFormat
from query_processing.domain.api_format import api_format_of
from query_processing.domain.routes import RouteChoice, RouteRequest


def test_the_route_is_named_route_on_the_wire() -> None:
    api_format = RouterApiFormat()
    wire = api_format.answer_to_wire(RouteChoice(name="a", confidence=0.5))
    assert wire == {"route": "a", "confidence": 0.5}
    assert api_format.answer_from_wire(wire) == RouteChoice(name="a", confidence=0.5)


def test_the_request_carries_the_vector_both_ways() -> None:
    api_format = RouterApiFormat()
    wire = api_format.request_to_wire(RouteRequest(embedding=[0.1, 0.2]))
    assert wire == {"embedding": [0.1, 0.2]}
    assert api_format.request_from_wire(wire) == RouteRequest(embedding=[0.1, 0.2])


def test_it_is_the_format_of_the_router() -> None:
    assert isinstance(api_format_of(BaseRouter), RouterApiFormat)


@pytest.mark.parametrize(
    "body", [{"name": "a", "confidence": 0.5}, {"route": "a"}, ["a", 0.5]]
)
def test_an_answer_of_another_shape_is_refused_naming_the_one_expected(
    body: object,
) -> None:
    with pytest.raises(ValueError, match=r"expected \{'route': <name>, 'confidence'"):
        RouterApiFormat().answer_from_wire(body)


def test_a_request_of_another_shape_is_refused_naming_the_one_expected() -> None:
    with pytest.raises(ValueError, match=r"expected \{'embedding': \[<float>"):
        RouterApiFormat().request_from_wire({"vector": [0.1]})
