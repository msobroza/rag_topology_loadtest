"""``ComponentResources``: what an implementation is handed to build itself.

An implementation of a component type builds itself with ``from_settings``,
from the block of the workflow file named after its backend. Anything else it
can need comes in this small object, and no inner part of the toolkit does:
the settings of its component type, the settings of another component by its
name, and for a router its routes. The composition fills it.

Example:
    >>> resources = ComponentResources(
    ...     name="anonymizer", type_settings=None,
    ...     component_settings=settings.components.__getitem__)
    >>> resources.component_settings("embedder").backend
    'local'
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class ComponentResources:
    """What an implementation's ``from_settings`` may use besides its own block.

    Example:
        >>> RegexAnonymizer.from_settings(block, resources).patterns
        {'email': re.compile(...)}
    """

    name: str  # the component's name in the workflow file
    # The keys of the component type, read with its settings class; None for a
    # type that has no keys of its own.
    type_settings: BaseModel | None
    # The settings of another component of the workflow file, by its name.
    component_settings: Callable[[str], BaseModel]
    # A router's routes, by name; empty for any other component.
    routes: Mapping[str, Any] = field(default_factory=dict)
