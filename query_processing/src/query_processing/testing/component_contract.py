"""``ComponentContract``: the tests a user runs on an implementation of theirs.

A user subclasses it in a test module and says how to make one component and
one request; their test runner collects the tests it inherits. They check that
the component's class names a backend, so that a workflow file can choose it;
that its type's request and answer are pydantic models; that the answer has
the shape of the component type; that the request and the answer survive the
type's API format; and that the type's ``served`` class answers as the
component does, through a model server run in process. A test that also gives
``settings_block``, its backend's block as the workflow file states it, has
the component built from it through ``from_settings``, as the app builds it.

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
from collections.abc import Callable, Mapping
from typing import Any, cast

import httpx
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from ..composition.composition_root import CompositionRoot
from ..composition.registered_components import SERVED, ComponentType
from ..config.workflow_settings import WorkflowSettings
from ..domain.components.base_component import BaseComponent
from ..domain.endpoints.api_endpoint import call_path
from ..domain.endpoints.api_format import api_format_of
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

    def component_keys(self) -> dict[str, object]:
        """Return the keys of its type the component's settings need, if any.

        Returns:
            No key by default; a component type with required keys of its own,
            such as a router's routes, gives them.

        Example:
            >>> TestFakeRouter().component_keys()
            {'embedder': 'embedder', 'routes': ['a', 'b']}
        """
        return {}

    def workflow_keys(self) -> dict[str, object]:
        """Return the top-level settings the component's settings need, if any.

        Returns:
            No setting by default; a router gives the routes it lists, as
            ``routes.yaml`` would, and under ``components`` the embedder it
            names, which is declared beside the component and never opened.

        Example:
            >>> list(TestFakeRouter().workflow_keys())
            ['routes', 'components']
        """
        return {}

    def settings_block(self) -> dict[str, object] | None:
        """Return the block of the workflow file that builds this component, if any.

        Returns:
            None by default: the component is then made by ``make_component``
            alone. A test that gives the block named after the backend, as the
            workflow file states it, has the component built from it through
            ``from_settings``, as the app builds it, and compared.

        Example:
            >>> TestUpperShouter().settings_block()
            {'suffix': '!'}
        """
        return None

    async def test_a_workflow_file_can_choose_its_backend(self) -> None:
        """The component's class names a backend of its component type.

        Raises:
            AssertionError: If it is registered under none; the message shows
                the class line to write and the backends the type has.

        Example:
            >>> await TestUpperShouter().test_a_workflow_file_can_choose_its_backend()
        """
        backend_of(self.make_component())

    async def test_its_request_and_answer_are_pydantic_models(self) -> None:
        """The component type's request and answer are pydantic models.

        Raises:
            AssertionError: If one is not; the message names both.

        Example:
            >>> contract = TestUpperShouter()
            >>> await contract.test_its_request_and_answer_are_pydantic_models()
        """
        port = component_type_of(self.make_component())
        request, answer = port.request_type, port.answer_type
        both = all(_is_pydantic_model(model) for model in (request, answer))
        assert both, (
            f"{port.__name__} takes {_named(request)} and answers {_named(answer)}"
            "; expected two pydantic models (class Request(BaseModel), class "
            "Answer(BaseModel)), which the toolkit reads fields from and carries "
            "as JSON"
        )

    async def test_it_builds_from_its_settings_block(self) -> None:
        """Built from ``settings_block`` as the app builds it, it answers alike.

        Nothing is checked when ``settings_block`` gives None.

        Raises:
            AssertionError: If the two answers differ; the message shows both.

        Example:
            >>> await TestUpperShouter().test_it_builds_from_its_settings_block()
        """
        block = self.settings_block()
        if block is None:
            return
        component, request = self.make_component(), self.make_request()
        expected = await component.ainvoke(request)
        built = await built_from_settings(
            component, block, self.component_keys(), self.workflow_keys()
        )
        answer = await built.ainvoke(request)
        assert answer == expected, (
            f"{type(built).__name__} built from its settings block {block!r} "
            f"answered {answer!r}; expected {expected!r}, the answer of the "
            "component make_component returns"
        )

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
        served = await served_answer(
            component, request, self.component_keys(), self.workflow_keys()
        )
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


def backend_of(component: object) -> str:
    """Return the backend a component's class registers under.

    Args:
        component: What ``make_component`` returned.

    Returns:
        The backend its class names on its class line.

    Raises:
        AssertionError: If its class names none; the message shows the class
            line to write and the backends its type has.

    Example:
        >>> backend_of(UpperShouter(suffix="!"))
        'local'
    """
    port, cls = component_type_of(component), type(component)
    for backend, implementation in port.backends.items():
        if implementation is cls:
            return backend
    has = ", ".join(repr(name) for name in sorted(port.backends)) or "none"
    # 'local' when it is free: a first implementation runs in process.
    name = "<its name>" if "local" in port.backends else "local"
    raise AssertionError(
        f"{cls.__name__} names no backend, so no workflow file can choose it "
        f"(expected class {cls.__name__}({port.__name__}, backend='{name}'), a "
        f"backend {port.__name__} does not have yet; it has: {has})"
    )


async def built_from_settings(
    component: BaseComponent[Any, Any],
    block: Mapping[str, object],
    component_keys: Mapping[str, object] | None = None,
    workflow_keys: Mapping[str, object] | None = None,
) -> BaseComponent[Any, Any]:
    """Build a component of the same class from its block, as the app builds it.

    The block is read with the settings class the class names, and the
    composition opens the component through ``from_settings``.

    Args:
        component: A component of the class to build.
        block: Its backend's block, as the workflow file states it.
        component_keys: The keys of its type its settings need; None for none.
        workflow_keys: The top-level settings they need; None for none.

    Returns:
        The component the composition opened.

    Example:
        >>> await built_from_settings(UpperShouter(suffix="!"), {"suffix": "?"})
        <UpperShouter ...>
    """
    name, backend = (
        component_type_of(component).component_type_name,
        backend_of(component),
    )
    entry = {**(component_keys or {}), "backend": backend, backend: dict(block)}
    settings = _ContractSettings(**_blocks(name, entry, dict(workflow_keys or {})))
    async with CompositionRoot(settings) as composition:
        opened = (await composition.workflow_dependencies()).components()
        return cast(BaseComponent[Any, Any], opened[name])


async def served_answer(
    component: BaseComponent[Any, Any],
    request: BaseModel,
    component_keys: Mapping[str, object] | None = None,
    workflow_keys: Mapping[str, object] | None = None,
) -> Any:
    """Serve the component in process; return what its ``served`` class answers.

    Args:
        component: The component the model server serves.
        request: What the served class is asked.
        component_keys: The keys of its type its settings need; None for none.
        workflow_keys: The top-level settings they need; None for none.

    Returns:
        The answer the type's ``served`` class reads from the call route.

    Example:
        >>> await served_answer(UpperShouter(suffix="!"), ShoutRequest(text="a"))
        Shouted(text='A!')
    """
    name = component_type_of(component).component_type_name
    keys, top = dict(component_keys or {}), dict(workflow_keys or {})
    given = _ContractSettings(**_blocks(name, {**keys, "backend": _GIVEN_BACKEND}, top))
    supplied: dict[str, Any] = {name: component}  # a port supplied by its name
    server = create_model_server(name, CompositionRoot(given, **supplied))
    with TestClient(server):  # runs its lifespan, which opens the component given
        transport = httpx.ASGITransport(app=server)
        url = _MODEL_SERVER_URL + call_path(name)
        served_block = {**keys, "backend": SERVED, "served": {"url": url}}
        calling = _ContractSettings(**_blocks(name, served_block, top))
        async with CompositionRoot(calling, transport=transport) as composition:
            opened = (await composition.workflow_dependencies()).components()
            served = cast(BaseComponent[Any, Any], opened[name])
            return await served.ainvoke(request)


def _blocks(
    name: str, component: dict[str, Any], top: dict[str, object]
) -> dict[str, Any]:
    # One component, of the type named after it, and one stage that calls it,
    # so that the composition opens it and nothing else; and the top-level
    # settings its settings need. The components those settings name, such as
    # a router's embedder, are declared beside it, and no stage calls them.
    stage = {"call": name, "reads": [], "writes": []}
    declared_beside = top.get("components")
    components = dict(declared_beside) if isinstance(declared_beside, Mapping) else {}
    components[name] = {"type": name, **component}
    return {**top, "components": components, "graph": {"stages": {name: stage}}}


def _is_pydantic_model(model: object) -> bool:
    return isinstance(model, type) and issubclass(model, BaseModel)


def _named(model: object) -> str:
    return str(getattr(model, "__name__", model))


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
