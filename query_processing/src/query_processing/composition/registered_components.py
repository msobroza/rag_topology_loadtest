"""A component whose type registers in ``COMPONENT_TYPES``, opened by its own class.

Such a component type, an extension's or one of the toolkit's made the same
way, has no opener: each implementation names its backend on its class line
and builds itself with ``from_settings``. The composition still finds the
class, hands it its block, already read with the settings class it names, and
gathers its resources (ADR 0008).

Example:
    >>> shouter = await open_registered("shouter", settings.components["shouter"],
    ...                                 resources)
    >>> (await shouter.ainvoke(ShoutRequest(text="a"))).text
    'A!'
"""

from __future__ import annotations

from typing import Any

from ..config.registered_component_settings import RegisteredComponentSettings
from ..domain.base_component import BaseComponent
from ..domain.component_resources import ComponentResources


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
