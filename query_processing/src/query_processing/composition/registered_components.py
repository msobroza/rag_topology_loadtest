"""A component whose type registers in ``COMPONENT_TYPES``, opened by its own class.

Such a component type, an extension's or one of the toolkit's made the same
way, has no opener: each implementation names its backend on its class line
and builds itself with ``from_settings``. The composition still finds the
class, hands it its block, already read with the settings class it names, and
gathers its resources (ADR 0008).

Every such type has a ``served`` backend with no code from the user: the class
that calls a model server of ours on the type's call route, built here from
``ServedCallMixin`` and the type's abstract class. A type that names a
``served`` class of its own has it built in the built one's place, alike.

Example:
    >>> shouter = await open_registered("shouter", settings.components["shouter"],
    ...                                 resources)
    >>> (await shouter.ainvoke(ShoutRequest(text="a"))).text
    'A!'
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from typing import Any

from ..config.registered_component_settings import RegisteredComponentSettings
from ..config.workflow_settings import WorkflowSettings
from ..domain.api_endpoint import SERVED_BACKEND
from ..domain.api_format import api_format_of
from ..domain.base_component import BaseComponent
from ..domain.component_resources import ComponentResources
from ..infrastructure.endpoint.open_endpoint_client import open_endpoint_client
from ..infrastructure.endpoint.served_call import ServedCallMixin
from .http_resources import HttpResources

# The backend every registered component type has, whether a class names it or not.
SERVED = SERVED_BACKEND
# The backend whose model runs in this process; a model server opens it.
LOCAL = "local"
ComponentType = type[BaseComponent[Any, Any]]


async def open_registered(
    name: str, component: RegisteredComponentSettings, resources: ComponentResources
) -> BaseComponent[Any, Any]:
    """Open a component with the implementation its backend names.

    Args:
        name: The component's name, which the resources are gathered for.
        component: Its settings, its backend's block already read.
        resources: What its implementation may use besides its block.

    Returns:
        The implementation, built from its block.

    Raises:
        ValueError: If no implementation of its type names its backend; the
            message lists those that do.

    Example:
        >>> await open_registered("shouter", shouter_settings, resources)
        <UpperShouter ...>
    """
    del name  # the resources name it; kept so every opener is called alike
    implementation = component.component_type().backend_class(component.backend)
    return implementation.from_settings(component.backend_block(), resources)


async def open_served(
    name: str, component: RegisteredComponentSettings, http: HttpResources
) -> BaseComponent[Any, Any]:
    """Open the class that calls the component's model server on its call route.

    Args:
        name: The component's name: every error names it, and its model server
            serves it under it.
        component: Its settings: the ``served`` block's URL, which the
            settings refuse when it is no URL, and its ``http`` block.
        http: The shared HTTP client, the default policy and the run tokens.

    Returns:
        The type's own ``served`` class if it names one, else the built one,
        each built with the endpoint client and the type's API format.

    Example:
        >>> await open_served("shouter", shouter_settings, http)
        <ServedShouter ...>
    """
    port = component.component_type()
    endpoint_client = open_endpoint_client(
        await http.client(),
        http.call_endpoint(name, component),
        component=name,
        tokens=http.run_tokens,
        policy=http.policy_of(component),
    )
    return served_class_of(port)(endpoint_client, api_format_of(port))


def served_class_of(port: ComponentType) -> type[Any]:
    """Return the class a component type's ``served`` backend is built with.

    Args:
        port: The component type's abstract class.

    Returns:
        The class the type registers under ``served``, else one the toolkit
        builds from ``ServedCallMixin`` and ``port``, registered nowhere.

    Raises:
        TypeError: If the class the type registers is no ``ServedCallMixin``.

    Example:
        >>> served_class_of(BaseShouter).__name__
        'ServedShouter'
    """
    registered = port.backends.get(SERVED)
    if registered is None:
        return _built_served_class(port)
    if not issubclass(registered, ServedCallMixin):
        # It is built as the built one is, with the endpoint client and the format.
        raise TypeError(
            f"{registered.__name__} names backend 'served' but does not subclass "
            f"ServedCallMixin (expected class {registered.__name__}(ServedCallMixin, "
            f"{port.__name__}, backend='served'))"
        )
    return registered


@cache  # one class per component type, so two components of a type share it
def _built_served_class(port: ComponentType) -> type[Any]:
    name = "Served" + port.__name__.removeprefix("Base")
    doc = f"Calls a model server of ours for a {port.component_type_name}."
    return type(name, (ServedCallMixin, port), {"__doc__": doc})


def registered_backend_class(port: ComponentType, backend: str) -> object:
    """Return what opens a registered component on ``backend``, refusing an unknown one.

    Args:
        port: The component type's abstract class.
        backend: The backend a component's settings state.

    Returns:
        The implementation that names ``backend``, or the served class.

    Raises:
        ValueError: If the type has no such backend; the message lists those
            it has, ``served`` included.

    Example:
        >>> registered_backend_class(BaseShouter, "served").__name__
        'ServedShouter'
    """
    if backend == SERVED:
        return served_class_of(port)
    if backend in port.backends:
        return port.backends[backend]
    raise ValueError(
        f"unknown {port.component_type_name} backend {backend!r} "
        f"(expected one of: {_listed({*port.backends, SERVED})})"
    )


def local_settings(
    name: str, component: RegisteredComponentSettings
) -> RegisteredComponentSettings:
    """Return the settings a model server opens a component with, in its process.

    Args:
        name: The component's name, which an error names.
        component: The component, as the workflow file states it.

    Returns:
        It unchanged, unless it is on ``served``: a model server is what that
        backend calls, so it is then read again on ``local``.

    Raises:
        ValueError: If it is on ``served`` and its type has no ``local``
            backend; the message names the variable and the backends it has.

    Example:
        >>> local_settings("shouter", served_shouter).backend
        'local'
    """
    if component.backend != SERVED:
        return component
    port = component.component_type()
    if LOCAL not in port.backends:
        variable = WorkflowSettings.variable_of("components", name, "backend")
        raise ValueError(
            f"{variable}='served', and a {port.component_type_name} has no "
            f"{LOCAL!r} backend for its model server to open in its process; "
            f"set the one it opens (one of: {_listed(port.backends)})"
        )
    entry = {**component.model_dump(), "backend": LOCAL}
    # Read with its own class, so a router keeps the keys of its type.
    return type(component).model_validate(entry)


def _listed(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in sorted(names)) or "none"
