"""The test class the toolkit ships, run on the test extension's ``shouter``.

``TestUpperShouter`` is what a use case writes for a component of its own:
pytest collects the three tests it inherits. The others show that a component
whose answer is not its type's fails them, naming what was expected.
"""

from __future__ import annotations

import asyncio

import pytest

from query_processing import extension_api
from query_processing.extension_api import BaseComponent
from query_processing.testing import ComponentContract
from shout_extension.shouting import BaseShouter, Shouted, ShoutRequest, UpperShouter
from shout_extension.whispering import LowerWhisperer


class ReturningADict(BaseShouter):
    """Answers with the answer's fields, not the answer."""

    def invoke(self, request: ShoutRequest) -> Shouted:
        return {"text": request.text}  # type: ignore[return-value]


class LoudShouted(Shouted):
    """A class of its own, which the API format, knowing ``Shouted``, loses."""


class ReturningASubclassOfItsAnswer(BaseShouter):
    """Answers with a ``LoudShouted``, which comes back a ``Shouted``."""

    def invoke(self, request: ShoutRequest) -> Shouted:
        return LoudShouted(text=request.text)


class TestUpperShouter(ComponentContract):
    def make_component(self) -> BaseShouter:
        return UpperShouter(suffix="!")

    def make_request(self) -> ShoutRequest:
        return ShoutRequest(text="a")


class ContractOf(ComponentContract):
    """The contract of one ready-made component, asked ``a``."""

    def __init__(self, component: BaseComponent[ShoutRequest, Shouted]) -> None:
        self.component = component

    def make_component(self) -> BaseComponent[ShoutRequest, Shouted]:
        return self.component

    def make_request(self) -> ShoutRequest:
        return ShoutRequest(text="a")


def test_the_extension_api_does_not_export_the_shipped_classes() -> None:
    # Every command imports extension_api with a use case's extensions: the
    # shipped classes would load the model server and the test client with it.
    assert not hasattr(extension_api, "ComponentContract")
    assert not hasattr(extension_api, "RouterContract")


def test_the_contract_catches_an_answer_of_the_wrong_shape() -> None:
    wrong = ContractOf(ReturningADict())

    with pytest.raises(AssertionError, match="expected a Shouted"):
        asyncio.run(wrong.test_the_answer_has_the_shape_of_its_component_type())


def test_the_contract_catches_an_answer_the_api_format_loses() -> None:
    lossy = ContractOf(ReturningASubclassOfItsAnswer())

    with pytest.raises(AssertionError, match=r"LoudShouted.* as Shouted\(text='a'\)"):
        asyncio.run(lossy.test_the_answer_survives_the_api_format())


def test_the_contract_asks_the_served_class_through_a_model_server() -> None:
    # The whisperer's served class marks what the model server answered, so
    # the answers differ only if the call went through it.
    marked = ContractOf(LowerWhisperer())

    with pytest.raises(AssertionError, match=r"Shouted\(text='a \(served\)'\)"):
        asyncio.run(marked.test_the_served_class_answers_as_the_component_does())
