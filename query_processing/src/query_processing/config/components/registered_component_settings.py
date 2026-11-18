"""The settings of a component whose type an extension defines, or the toolkit's own.

A component type registered in ``COMPONENT_TYPES`` that has no settings class
of its own in ``COMPONENT_TYPE_SETTINGS`` is read with
``RegisteredComponentSettings``. Its keys are the ones every component has
(``type``, ``backend``, ``http``, ``served``), the keys of its component type
(read with the class the type names on its class line, ``settings=``), and one
block per backend, named after it and read with the settings class its
implementation names. A key that is none of these is refused, and so is a key
of a backend's block that its settings class does not have, each naming the
keys expected.

Example:
    >>> shouter = RegisteredComponentSettings.model_validate(
    ...     {"type": "shouter", "backend": "local", "local": {"suffix": "!"}})
    >>> shouter.backend_block().suffix
    '!'
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...domain.components.base_component import COMPONENT_TYPES, BaseComponent
from .settings_blocks import ComponentHttpSettings, ComponentSettings, ServedSettings

ComponentType = type[BaseComponent[Any, Any]]


class RegisteredComponentSettings(ComponentSettings):
    """A component of a type registered in ``COMPONENT_TYPES``: its blocks, read.

    The keys of its type and the block of each backend it states are kept
    beside its fields, each block already read with its implementation's
    settings class.

    Example:
        >>> settings.components["shouter"].backend_block()
        UpperShouterSettings(suffix='!')
    """

    # Its type's keys and its backends' blocks are known only once the type is:
    # the before validator refuses any other key, so none is kept by mistake.
    model_config = ConfigDict(frozen=True, extra="allow")

    http: ComponentHttpSettings = Field(default_factory=ComponentHttpSettings)
    served: ServedSettings = Field(default_factory=ServedSettings)

    @model_validator(mode="before")
    @classmethod
    def refuse_unknown_keys(cls, data: Any) -> Any:
        """Refuse a key its type does not have, and read each backend's block.

        Args:
            data: The component's entry, as read, its ``type`` given.

        Returns:
            ``data``, its type's keys and each backend's block read with their
            settings classes.

        Raises:
            ValueError: If the type is not registered, a key is not one of the
                component's, or a backend's block has a key its settings class
                does not have; the message names the keys expected.
        """
        if not isinstance(data, Mapping):
            return data
        component_type = COMPONENT_TYPES.get(str(data.get("type")))
        _refuse_unknown(cls, component_type, data)
        return {
            **data,
            **_type_keys(component_type, data),
            **_backend_blocks(component_type, data),
        }

    def component_type(self) -> ComponentType:
        """Return the component type, the abstract class registered under ``type``.

        Returns:
            The class whose implementations ``backend`` picks from.

        Example:
            >>> settings.components["shouter"].component_type()
            <class 'BaseShouter'>
        """
        return COMPONENT_TYPES.get(self.type)

    def backend_block(self) -> BaseModel | None:
        """Return the block of the backend in use, read with its settings class.

        Returns:
            The block, its defaults filled when the file leaves it out; None
            when the implementation names no settings class, or no
            implementation names the backend (``check_registered_names``
            refuses that).

        Example:
            >>> settings.components["shouter"].backend_block().suffix
            '!'
        """
        block = (self.model_extra or {}).get(self.backend)
        return block if isinstance(block, BaseModel) else None

    def type_block(self) -> BaseModel | None:
        """Return the keys of the component type, read with the type's settings class.

        Returns:
            Them, defaults filled; None for a type that has no keys of its own.

        Example:
            >>> settings.components["router"].type_block().embedder
            'embedder'
        """
        type_settings = self.component_type().type_settings
        if type_settings is None:
            return None
        extra = self.model_extra or {}
        keys = {k: v for k, v in extra.items() if k in type_settings.model_fields}
        return type_settings.model_validate(keys)


def _refuse_unknown(
    cls: type[ComponentSettings], component_type: ComponentType, data: Mapping[Any, Any]
) -> None:
    type_keys = sorted(_fields(component_type.type_settings))
    backends = sorted(component_type.backends)
    known = {*cls.model_fields, *type_keys, *backends}
    unknown = sorted(str(key) for key in data if key not in known)
    if unknown:
        raise ValueError(
            f"the {component_type.component_type_name!r} component type has no "
            f"{', '.join(map(repr, unknown))}; expected its keys "
            f"({', '.join(sorted({*cls.model_fields, *type_keys}))}), and a "
            f"backend's settings in the block named after it ({_listed(backends)})"
        )


def _type_keys(
    component_type: ComponentType, data: Mapping[Any, Any]
) -> dict[str, Any]:
    # The keys of the type that the entry states, each read with its class.
    type_settings = component_type.type_settings
    if type_settings is None:
        return {}
    given = {k: v for k, v in data.items() if k in type_settings.model_fields}
    read = type_settings.model_validate(given)
    return {key: getattr(read, key) for key in given}


def _backend_blocks(
    component_type: ComponentType, data: Mapping[Any, Any]
) -> dict[str, BaseModel]:
    # Each block the entry states, and the one of the backend in use whether it
    # is stated or not, so a block with a required key cannot be left out. A
    # served class a type registers reads the component's own served block.
    in_use, backends = data.get("backend"), component_type.backends
    named = [
        backend
        for backend in backends
        if (backend in data or backend == in_use)
        and backend not in RegisteredComponentSettings.model_fields
    ]
    read = {
        name: _read_block(component_type, name, backends[name], data.get(name))
        for name in named
    }
    return {backend: block for backend, block in read.items() if block is not None}


def _read_block(
    component_type: ComponentType,
    backend: str,
    implementation: ComponentType,
    block: object,
) -> BaseModel | None:
    entries = _as_mapping(block)
    settings_type = implementation.settings_type
    keys = _fields(settings_type)
    unknown = sorted(str(key) for key in entries if key not in keys)
    if unknown:
        raise ValueError(
            f"the {backend!r} block of a {component_type.component_type_name!r} "
            f"component has no {', '.join(map(repr, unknown))}; expected its keys "
            f"({_listed(sorted(keys))}), those of {implementation.__name__}"
        )
    return None if settings_type is None else settings_type.model_validate(entries)


def _as_mapping(block: object) -> Mapping[Any, Any]:
    if block is None:
        return {}
    if isinstance(block, BaseModel):  # settings read already, read again
        return block.model_dump()
    if isinstance(block, Mapping):
        return block
    raise ValueError(
        f"a backend's block is {block!r}; expected a mapping of its keys, "
        "such as {suffix: '!'}"
    )


def _fields(settings_type: type[BaseModel] | None) -> dict[str, Any]:
    return {} if settings_type is None else dict(settings_type.model_fields)


def _listed(names: list[str]) -> str:
    return ", ".join(names) or "none"
