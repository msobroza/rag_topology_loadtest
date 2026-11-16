"""``ComponentContract``: the tests a user runs on an implementation of theirs.

A user subclasses it in a test module and says how to make one component and
one request; their test runner collects the three tests it inherits. They
check that the answer has the shape of the component type, that the request
and the answer survive the type's API format, and that the type's ``served``
class answers as the component does, through a model server run in process.

The tests are coroutines, as the toolkit's own async tests are: a runner that
awaits them is needed, such as pytest with pytest-asyncio's ``asyncio_mode =
"auto"``. Nothing here imports pytest. The model server is served from memory,
with no socket, and the component given is the one it serves: the user's
workflow file and ``QP_*`` variables are not read.

Example:
    >>> class TestUpperShouter(ComponentContract):
    ...     def make_component(self) -> BaseShouter:
    ...         return UpperShouter(suffix="!")
    ...     def make_request(self) -> ShoutRequest:
    ...         return ShoutRequest(text="a")
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import httpx
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from ..composition.composition_root import CompositionRoot
from ..composition.registered_components import SERVED, ComponentType
from ..config.workflow_settings import WorkflowSettings
from ..domain.api_endpoint import call_path
from ..domain.api_format import api_format_of
from ..domain.base_component import BaseComponent
from ..entrypoints.qp_model_server import create_model_server

# The backend the model server's settings name: none of the type's, so that no
# implementation's block is read, and the component given is the one opened.
_GIVEN_BACKEND = "given"
# The served class's URL: the in-process transport reaches the app whatever the host.
_MODEL_SERVER_URL = "http://model-server"


class ComponentContract(ABC):
    """The tests every implementation of a component type passes.

    Subclass it in a test module, under a name the test runner collects, and
    write ``make_component`` and ``make_request``. The component's type must
    be registered on its class line, as ``extension_api`` shows.

    Example:
        >>> class TestRegexAnonymizer(ComponentContract):
        ...     def make_component(self) -> BaseAnonymizer:
        ...         return RegexAnonymizer(mask="*")
        ...     def make_request(self) -> AnonymizeRequest:
        ...         return AnonymizeRequest(text="Ana Lima")
    """

    @abstractmethod
    def make_component(self) -> BaseComponent[Any, Any]:
        """Return a new component of the implementation under test.

        Returns:
            The component, ready to answer.

        Example:
            >>> TestUpperShouter().make_component()
            <UpperShouter ...>
        """

    @abstractmethod
    def make_request(self) -> BaseModel:
        """Return a request of the component's type, the one every test sends.

        Returns:
            The request.

        Example:
            >>> TestUpperShouter().make_request()
            ShoutRequest(text='a')
        """

    async def test_the_answer_has_the_shape_of_its_component_type(self) -> None:
        """The component answers with its component type's answer class.

        Raises:
            AssertionError: If the answer is of another class; the message
                names the class expected.

        Example:
            >>> contract = TestUpperShouter()
            >>> await contract.test_the_answer_has_the_shape_of_its_component_type()
        """
        component = self.make_component()
        answer = await component.ainvoke(self.make_request())
        expected = component_type_of(component).answer_type
        assert isinstance(answer, expected), (
            f"{type(component).__name__} answered {answer!r}, a "
            f"{type(answer).__name__}; expected a {expected.__name__}"
        )

    async def test_the_answer_survives_the_api_format(self) -> None:
        """The request and the answer come back from the wire as they went.

        Raises:
            AssertionError: If one comes back changed, or its API format
                cannot read it back; the message shows both.

        Example:
            >>> await TestUpperShouter().test_the_answer_survives_the_api_format()
        """
        component = self.make_component()
        api_format = api_format_of(component_type_of(component))
        request = self.make_request()
        answer = await component.ainvoke(request)
        _assert_survives(
            request, api_format.request_to_wire, api_format.request_from_wire
        )
        _assert_survives(answer, api_format.answer_to_wire, api_format.answer_from_wire)

    async def test_the_served_class_answers_as_the_component_does(self) -> None:
        """The type's ``served`` class, calling a model server of it, answers alike.

        Raises:
            AssertionError: If the two answers differ; the message shows both.

        Example:
            >>> contract = TestUpperShouter()
            >>> await contract.test_the_served_class_answers_as_the_component_does()
        """
        component = self.make_component()
        request = self.make_request()
        expected = await component.ainvoke(request)
        served = await served_answer(component, request)
        assert served == expected, (
            f"the served {type(component).__name__} answered {served!r}; expected "
            f"{expected!r}, what it answers in process"
        )


def component_type_of(component: object) -> ComponentType:
    """Return the component type a component implements: its abstract class.

    Args:
        component: What ``make_component`` returned.

    Returns:
        The class of its ancestry defined with ``component_type=``.

    Raises:
        TypeError: If it implements no component type.

    Example:
        >>> component_type_of(UpperShouter(suffix="!"))
        <class 'BaseShouter'>
    """
    for cls in type(component).__mro__:
        if issubclass(cls, BaseComponent) and "component_type_name" in vars(cls):
            return cls
    raise TypeError(
        f"make_component returned {component!r}, a {type(component).__name__}; "
        "expected an implementation of a component type (a subclass of a class "
        "defined with BaseComponent[Request, Answer], component_type=...)"
    )


async def served_answer(component: BaseComponent[Any, Any], request: BaseModel) -> Any:
    """Serve the component in process; return what its ``served`` class answers.

    Args:
        component: The component the model server serves.
        request: What the served class is asked.

    Returns:
        The answer the type's ``served`` class reads from the call route.

    Example:
        >>> await served_answer(UpperShouter(suffix="!"), ShoutRequest(text="a"))
        Shouted(text='A!')
    """
    name = component_type_of(component).component_type_name
    given = _ContractSettings(**_blocks(name, {"backend": _GIVEN_BACKEND}))
    supplied: dict[str, Any] = {name: component}  # a port supplied by its name
    server = create_model_server(name, CompositionRoot(given, **supplied))
    with TestClient(server):  # runs its lifespan, which opens the component given
        transport = httpx.ASGITransport(app=server)
        url = _MODEL_SERVER_URL + call_path(name)
        calling = _ContractSettings(
            **_blocks(name, {"backend": SERVED, "served": {"url": url}})
        )
        async with CompositionRoot(calling, transport=transport) as composition:
            opened = (await composition.workflow_dependencies()).components()
            served = cast(BaseComponent[Any, Any], opened[name])
            return await served.ainvoke(request)


def _blocks(name: str, component: dict[str, Any]) -> dict[str, Any]:
    # One component, of the type named after it, and one stage that calls it,
    # so that the composition opens it and nothing else.
    stage = {"call": name, "reads": [], "writes": []}
    return {
        "components": {name: {"type": name, **component}},
        "graph": {"stages": {name: stage}},
    }


def _assert_survives(
    value: Any, to_wire: Callable[[Any], object], from_wire: Callable[[object], Any]
) -> None:
    body = json.loads(json.dumps(to_wire(value)))  # as the wire carries it
    try:
        back = from_wire(body)
    except ValueError as exc:
        raise AssertionError(
            f"{value!r} goes on the wire as {body!r}, which its API format cannot "
            f"read back: {exc}"
        ) from exc
    assert (
        back == value
    ), f"{value!r} comes back from the wire as {back!r}; expected it unchanged"


class _ContractSettings(WorkflowSettings):
    # Only the blocks the contract gives: the user's workflow file, QP_*
    # variables and .env decide nothing about how it serves their component.
    # Pydantic passes every source by keyword, so the names stay.
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls, env_settings, dotenv_settings, file_secret_settings
        return (init_settings,)
