"""A component type: one abstract class that names its request and its answer."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from query_processing.domain.base_component import COMPONENT_TYPES, BaseComponent
from query_processing.domain.component_resources import ComponentResources
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


def test_settings_named_without_from_settings_are_refused_showing_it() -> None:
    # Built with no argument, it would answer with its constructor's defaults,
    # its block of the workflow file read and then ignored.
    def define_forgetful() -> type:
        class Forgetful(BaseShouter, backend="other", settings=UpperShouterSettings):
            def invoke(self, request: ShoutRequest) -> Shouted:
                return Shouted(text=request.text)

        return Forgetful

    with pytest.raises(TypeError) as refused:
        define_forgetful()

    assert str(refused.value) == (
        "Forgetful names settings=UpperShouterSettings but writes no "
        "from_settings, so its block of the workflow file would be read and never "
        "used (expected, in Forgetful:\n"
        "    @classmethod\n"
        "    def from_settings(cls, settings: UpperShouterSettings, "
        "resources: ComponentResources) -> Forgetful: ...)"
    )
    assert "other" not in BaseShouter.backends


@pytest.mark.parametrize(
    ("written", "found"),
    [
        (classmethod(lambda cls, settings: cls()), "(cls, settings)"),
        (lambda self, settings, resources: self, "(self, settings, resources)"),
    ],
    ids=["no resources", "not a classmethod"],
)
def test_a_from_settings_the_toolkit_cannot_call_is_refused_showing_it(
    written: object, found: str
) -> None:
    def define_wrong() -> type:
        def invoke(self: object, request: ShoutRequest) -> Shouted:
            return Shouted(text=request.text)

        namespace = {"from_settings": written, "invoke": invoke}
        return type(BaseShouter)(
            "Wrong",
            (BaseShouter,),
            namespace,
            backend="other",
            settings=UpperShouterSettings,
        )

    with pytest.raises(TypeError) as refused:
        define_wrong()

    assert str(refused.value) == (
        f"Wrong.from_settings{found} is not what the toolkit calls with the "
        "block of the workflow file and the component's resources (expected, in "
        "Wrong:\n"
        "    @classmethod\n"
        "    def from_settings(cls, settings: UpperShouterSettings, "
        "resources: ComponentResources) -> Wrong: ...)"
    )
    assert "other" not in BaseShouter.backends


def test_a_from_settings_written_without_settings_named_is_accepted() -> None:
    # The fake router reads its routes from its resources, and names no block.
    class FromResources(BaseShouter, backend="from-resources"):
        @classmethod
        def from_settings(
            cls, settings: BaseModel | None, resources: ComponentResources
        ) -> FromResources:
            del settings, resources
            return cls()

        def invoke(self, request: ShoutRequest) -> Shouted:
            return Shouted(text=request.text)

    assert BaseShouter.backends.pop("from-resources") is FromResources


def test_a_request_or_an_answer_that_is_no_pydantic_model_is_refused() -> None:
    @dataclass
    class Plain:
        text: str

    types: Registry[type[BaseComponent[Any, Any]]] = Registry("component type")

    def define_plain() -> type:
        class BasePlain(
            BaseComponent[ShoutRequest, Plain], component_type="plain", registry=types
        ): ...

        return BasePlain

    with pytest.raises(TypeError) as refused:
        define_plain()

    assert str(refused.value) == (
        "component type BasePlain answers a Plain, which is no pydantic model: "
        "the toolkit reads the workflow state's fields from a request's and an "
        "answer's fields, and carries them as JSON to a model server (expected "
        "class Plain(BaseModel), with its fields as annotations)"
    )
    assert types.names() == ()


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
