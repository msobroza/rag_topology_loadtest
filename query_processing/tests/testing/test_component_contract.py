"""The test class the toolkit ships, run on the test extension's ``shouter``.

``TestUpperShouter`` is what a use case writes for a component of its own:
pytest collects the tests it inherits. The others show that a component
whose answer is not its type's fails them, naming what was expected.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel
from query_processing_core.retrieval.metadata_filter import MetadataCondition

from query_processing import extension_api
from query_processing.adapters.llm.base_llm import BaseLLM
from query_processing.adapters.llm.fake.fake_llm import FakeLLM
from query_processing.domain.components.component_requests import (
    GenerateAnswer,
    GenerateRequest,
)
from query_processing.domain.errors import RagDependencyError
from query_processing.domain.registry import Registry
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

    def settings_block(self) -> dict[str, object]:
        return {"suffix": "!"}


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


def test_the_extension_api_exports_what_a_use_case_names_in_its_code() -> None:
    # An output schema that holds a metadata filter names MetadataCondition;
    # a component whose dependency fails raises the error the app answers
    # with 503 or 504.
    assert extension_api.MetadataCondition is MetadataCondition
    assert extension_api.RagDependencyError is RagDependencyError
    assert {"MetadataCondition", "RagDependencyError"} <= set(extension_api.__all__)


def test_the_extension_api_exports_what_a_use_cases_llm_names() -> None:
    # A use case's LLM subclasses BaseLLM and answers a GenerateAnswer.
    assert extension_api.BaseLLM is BaseLLM
    assert extension_api.GenerateAnswer is GenerateAnswer
    assert extension_api.GenerateRequest is GenerateRequest
    exported = {"BaseLLM", "GenerateAnswer", "GenerateRequest"}
    assert exported <= set(extension_api.__all__)


def test_the_contract_catches_an_answer_of_the_wrong_shape() -> None:
    wrong = ContractOf(ReturningADict())

    with pytest.raises(AssertionError, match="expected a Shouted"):
        asyncio.run(wrong.test_the_answer_has_the_shape_of_its_component_type())


class AnsweringText(BaseLLM):
    """An LLM written to the port of before: ``generate`` returns its text."""

    model_name = "text-llm"

    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> GenerateAnswer:
        return "Paris."  # type: ignore[return-value]


class LLMContractOf(ComponentContract):
    """The contract of one ready-made LLM, asked one message."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def make_component(self) -> BaseLLM:
        return self.llm

    def make_request(self) -> GenerateRequest:
        return GenerateRequest(messages=[{"role": "user", "content": "Capital?"}])


def test_the_contract_catches_an_llm_that_answers_bare_text() -> None:
    wrong = LLMContractOf(AnsweringText())

    with pytest.raises(AssertionError, match="expected a GenerateAnswer"):
        asyncio.run(wrong.test_the_answer_has_the_shape_of_its_component_type())


def test_an_llm_answer_and_its_token_counts_survive_the_api_format() -> None:
    contract = LLMContractOf(FakeLLM())

    asyncio.run(contract.test_the_answer_has_the_shape_of_its_component_type())
    asyncio.run(contract.test_the_answer_survives_the_api_format())


def test_the_contract_catches_an_answer_the_api_format_loses() -> None:
    lossy = ContractOf(ReturningASubclassOfItsAnswer())

    with pytest.raises(AssertionError, match=r"LoudShouted.* as Shouted\(text='a'\)"):
        asyncio.run(lossy.test_the_answer_survives_the_api_format())


def test_the_contract_catches_a_class_that_names_no_backend() -> None:
    # No workflow file can choose it: it is registered under no backend.
    unnamed = ContractOf(ReturningADict())

    with pytest.raises(AssertionError) as caught:
        asyncio.run(unnamed.test_a_workflow_file_can_choose_its_backend())

    assert str(caught.value).startswith(
        "ReturningADict names no backend, so no workflow file can choose it "
        "(expected class ReturningADict(BaseShouter, backend='<its name>'), a backend "
        "BaseShouter does not have yet; it has: 'fake', 'local')"
    )


def test_the_contract_catches_a_request_or_an_answer_that_is_no_pydantic_model() -> (
    None
):
    # Any cannot be read when the type is defined: the contract reads it.
    types: Registry[type[BaseComponent[Any, Any]]] = Registry("component type")

    class BaseLoose(
        BaseComponent[Any, Any], component_type="loose", registry=types
    ): ...

    class Loose(BaseLoose, backend="local"):
        def invoke(self, request: Any) -> Any:
            return request

    with pytest.raises(AssertionError) as caught:
        asyncio.run(
            ContractOf(Loose()).test_its_request_and_answer_are_pydantic_models()
        )

    assert str(caught.value).startswith(
        "BaseLoose takes Any and answers Any; expected two pydantic models"
    )


def test_the_contract_builds_the_component_from_its_settings_block() -> None:
    # The block gives "?", the component made by hand "!": built through
    # from_settings, as the app builds it, it answers otherwise.
    class BuiltOtherwise(TestUpperShouter):
        def settings_block(self) -> dict[str, object]:
            return {"suffix": "?"}

    with pytest.raises(AssertionError) as caught:
        asyncio.run(BuiltOtherwise().test_it_builds_from_its_settings_block())

    assert str(caught.value) == (
        "UpperShouter built from its settings block {'suffix': '?'} answered "
        "Shouted(text='A?'); expected Shouted(text='A!'), the answer of the "
        "component make_component returns"
    )


def test_the_contract_asks_the_served_class_through_a_model_server() -> None:
    # The whisperer's served class marks what the model server answered, so
    # the answers differ only if the call went through it.
    marked = ContractOf(LowerWhisperer())

    with pytest.raises(AssertionError, match=r"Shouted\(text='a \(served\)'\)"):
        asyncio.run(marked.test_the_served_class_answers_as_the_component_does())
