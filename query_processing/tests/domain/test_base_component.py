"""A component type: one abstract class that names its request and its answer."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from pydantic import BaseModel

from query_processing.domain.base_component import BaseComponent
from query_processing.domain.registry import Registry


class EchoRequest(BaseModel):
    text: str


class Echo(BaseModel):
    text: str
    length: int


ECHO_TYPES: Registry[type[BaseComponent[Any, Any]]] = Registry("component type")


class BaseEcho(
    BaseComponent[EchoRequest, Echo], component_type="echo", registry=ECHO_TYPES
):
    """Answers a text with what it found in it."""


def test_a_component_type_registers_and_reads_its_request_and_answer() -> None:
    assert ECHO_TYPES.get("echo") is BaseEcho
    assert (BaseEcho.request_type, BaseEcho.answer_type) == (EchoRequest, Echo)
    assert BaseEcho.component_type_name == "echo"


async def test_ainvoke_runs_invoke_off_the_event_loop() -> None:
    class Upper(BaseEcho):
        def invoke(self, request: EchoRequest) -> Echo:
            return Echo(text=request.text.upper(), length=threading.get_ident())

    answer = await Upper().ainvoke(EchoRequest(text="a"))

    assert answer.text == "A"
    assert answer.length != threading.get_ident()


async def test_an_implementation_that_writes_neither_method_says_what_to_write() -> (
    None
):
    class Empty(BaseEcho):
        pass

    with pytest.raises(NotImplementedError, match=r"Empty writes no invoke.*-> Echo"):
        await Empty().ainvoke(EchoRequest(text="a"))


def test_a_component_type_that_names_no_request_and_answer_is_refused() -> None:
    types: Registry[type[BaseComponent[Any, Any]]] = Registry("component type")

    def define_bare() -> type:
        class Bare(BaseComponent, component_type="bare", registry=types):  # type: ignore[type-arg]
            pass

        return Bare

    with pytest.raises(TypeError, match=r"Bare names no request and answer"):
        define_bare()
    assert types.names() == ()
