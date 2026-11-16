"""A component type: one abstract class that names its request and its answer."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from pydantic import BaseModel

from query_processing.domain.base_component import COMPONENT_TYPES, BaseComponent
from query_processing.domain.registry import Registry
from shout_components import resources_for
from shout_extension.shouting import (
    BaseShouter,
    FakeShouter,
    Shouted,
    ShoutRequest,
    UpperShouter,
    UpperShouterSettings,
)


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


# --- an implementation names its backend and its settings on its class line ------


def test_an_implementation_registers_under_its_backend_with_its_settings() -> None:
    assert BaseShouter.backends == {"local": UpperShouter, "fake": FakeShouter}
    assert UpperShouter.settings_type is UpperShouterSettings
    assert FakeShouter.settings_type is None
    assert COMPONENT_TYPES.get("shouter") is BaseShouter


def test_an_implementation_builds_itself_from_its_settings() -> None:
    shouter = BaseShouter.backend_class("local").from_settings(
        UpperShouterSettings(suffix="!"), resources_for("shouter")
    )

    assert shouter.invoke(ShoutRequest(text="a")) == Shouted(text="A!")


def test_an_implementation_with_no_settings_builds_itself_with_none() -> None:
    shouter = FakeShouter.from_settings(None, resources_for("shouter"))

    assert isinstance(shouter, FakeShouter)


def test_a_backend_the_component_type_does_not_have_lists_those_it_has() -> None:
    with pytest.raises(
        ValueError,
        match=r"unknown shouter backend 'loud' \(expected one of: 'fake', 'local'\)",
    ):
        BaseShouter.backend_class("loud")


def test_a_backend_with_no_method_written_is_refused_when_defined() -> None:
    with pytest.raises(
        TypeError, match=r"Empty still has to write: invoke\(self, request\) -> Shouted"
    ):

        class Empty(BaseShouter, backend="other"): ...

    assert "other" not in BaseShouter.backends


def test_a_backend_named_twice_is_refused_and_names_both() -> None:
    def define_second() -> type:
        class Second(BaseShouter, backend="local"):
            def invoke(self, request: ShoutRequest) -> Shouted:
                return Shouted(text=request.text)

        return Second

    with pytest.raises(
        TypeError, match=r"shouter backend 'local' is already UpperShouter.*Second"
    ):
        define_second()
    assert BaseShouter.backends["local"] is UpperShouter


def test_a_class_that_names_no_backend_is_neither_registered_nor_refused() -> None:
    class Helper(BaseShouter):
        """A base the implementations of a use case could share."""

    assert Helper not in BaseShouter.backends.values()


def test_settings_named_without_a_backend_are_refused() -> None:
    def define_loose() -> type:
        class Loose(BaseShouter, settings=UpperShouterSettings): ...

        return Loose

    with pytest.raises(TypeError, match=r"Loose names settings but no backend"):
        define_loose()


def test_a_class_that_is_both_a_type_and_an_implementation_is_refused() -> None:
    types: Registry[type[BaseComponent[Any, Any]]] = Registry("component type")

    def define_both() -> type:
        class Both(
            BaseComponent[ShoutRequest, Shouted],
            component_type="both",
            registry=types,
            backend="local",
        ): ...

        return Both

    with pytest.raises(TypeError, match=r"Both names both component_type 'both'"):
        define_both()
    assert types.names() == ()
