"""The ``key`` blocks as read, checked before the settings library sees them.

The settings library quotes the value of whatever it refuses, and a person who
types an API key in the wrong place would see it printed. So every ``key``
block, and the old ``api_key`` anywhere under a component, is checked here on
the values as read, from every place a setting comes from merged, and each
refusal is one line that names the place by its path and its variable, never a
value: ``check_key_block`` is that one check of a block, and
``refuse_key_mistakes`` walks the ``components`` block with it.
``KeyBlocksChecked`` runs them as the last settings source, on what the
sources before it gave (pydantic-settings' public ``current_state``).

The places a setting comes from are, highest first, the values a command gives,
the environment, ``.env``, then the workflow file. A ``key`` block that a place
above the file gives with a ``source`` replaces the file's block whole
(``file_key_blocks_given_above_left_out``): a site can change where a key comes
from without editing the file, and the file's ``variable`` is not kept beside a
``domino`` source. A place above that sets another key alone changes that key.

Example:
    >>> check_key_block({"source": "env", "variable": "1X"},
    ...     variable="QP_LOADTEST_KEY", place="key")
    ValueError: QP_LOADTEST_KEY__VARIABLE (key.variable) is not a name: ...
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from .component_settings import COMPONENT_TYPE_SETTINGS

_OLD_KEY = "api_key"  # the key that held the key itself before key sources
_KEY = "key"
_SOURCE = "source"
# The keys of a block that name something, and what they name.
_NAMES = {_SOURCE: "a key source's name", "variable": "an environment variable's name"}
_EXAMPLES = (
    "{source: env, variable: <the variable that holds the key>} or {source: domino}"
)
VariableOf = Callable[..., str]  # WorkflowSettings.variable_of


def check_key_block(block: object, *, variable: str, place: str) -> None:
    """Refuse a ``key`` block not shaped as one, in a line that quotes no value.

    The one check of a block before the settings library reads it: a block
    that is not a mapping, a block with no ``source``, a value beside it that
    is not text, a ``source`` or a ``variable`` not shaped like a name
    (letters, digits and underscores, not starting with a digit). None is no
    block. Whether its source is registered and reads its keys is start-up's
    check (``checked_key_source``).

    Args:
        block: The block as read, any shape.
        variable: The variable it is read from, such as
            ``QP_COMPONENTS__LLM__API__KEY`` or ``QP_LOADTEST_KEY``.
        place: Its path, such as ``components.llm.api.key``.

    Raises:
        ValueError: The first mistake, naming the variable and the place.

    Example:
        >>> check_key_block("sk-1", variable="QP_LOADTEST_KEY", place="key")
        ValueError: QP_LOADTEST_KEY (key) is not a key block: ...
    """
    if block is None:
        return
    if not isinstance(block, Mapping):
        raise ValueError(
            f"{variable} ({place}) is not a key block: expected a mapping such as "
            f"{_EXAMPLES}, never the key itself"
        )
    if _SOURCE not in block:
        raise ValueError(
            f"{variable}__SOURCE is unset: the key block {place} names no key "
            f"source (expected {_EXAMPLES})"
        )
    for key, value in block.items():
        _check_block_value(str(key), value, variable, place)


def _check_block_value(key: str, value: object, variable: str, place: str) -> None:
    where = f"{variable}__{key.upper()} ({place}.{key})"
    if not isinstance(value, str):
        raise ValueError(f"{where} is a {type(value).__name__}: expected text")
    if key in _NAMES and not is_a_name(value):
        raise ValueError(
            f"{where} is not a name: expected {_NAMES[key]}, of letters, digits "
            "and underscores, not starting with a digit"
        )


def is_a_name(text: str) -> bool:
    """Say whether ``text`` is shaped like a name: ASCII letters, digits, underscores.

    Args:
        text: A source's or a variable's name as given.

    Returns:
        True when it is not empty and does not start with a digit.

    Example:
        >>> is_a_name("EXAMPLE_LLM_KEY"), is_a_name("sk-1")
        (True, False)
    """
    return text.isascii() and text.isidentifier()


def refuse_key_mistakes(components: object, variable_of: VariableOf) -> None:
    """Refuse an old ``api_key`` under a component, and a ``key`` block misshapen.

    Args:
        components: A ``components`` block as read, any shape: a mapping of
            names to entries is checked, anything else passes.
        variable_of: Names the variable a value is read from.

    Raises:
        ValueError: The first mistake, in one line that quotes no value.

    Example:
        >>> refuse_key_mistakes({"llm": {"served": {"api_key": "x"}}}, variable_of)
        ValueError: QP_COMPONENTS__LLM__SERVED__API_KEY (components.llm.served ...
    """
    for name, entry in _mappings(components):
        held = _old_key_holder(entry)
        if held is not None:
            raise ValueError(_old_api_key_message(name, held, entry, variable_of))
        _check_key_blocks_of(name, entry, variable_of)


def _check_key_blocks_of(
    name: str, entry: Mapping[str, Any], variable_of: VariableOf
) -> None:
    for block_name, block in _mappings(entry):
        if _KEY in block:
            at = ("components", name, block_name, _KEY)
            check_key_block(block[_KEY], variable=variable_of(*at), place=".".join(at))


def _mappings(value: object) -> list[tuple[str, Mapping[str, Any]]]:
    # The entries of a mapping that are mappings, by name; none of anything else.
    if not isinstance(value, Mapping):
        return []
    return [(str(k), v) for k, v in value.items() if isinstance(v, Mapping)]


def _old_key_holder(entry: Mapping[str, Any]) -> tuple[str, ...] | None:
    # The path under the component of the block holding api_key: () for itself.
    if _OLD_KEY in entry:
        return ()
    held = [(name,) for name, block in _mappings(entry) if _OLD_KEY in block]
    return held[0] if held else None


def _old_api_key_message(
    name: str, held: tuple[str, ...], entry: Mapping[str, Any], variable_of: VariableOf
) -> str:
    old = variable_of("components", name, *held, _OLD_KEY)
    path = ".".join(("components", name, *held, _OLD_KEY))
    said = f"{old} ({path} in a workflow file) is no longer read: "
    keyed = _block_taking_a_key(name, entry, held)
    if keyed is None:
        return said + (
            f"the {name} component takes no API key of its settings (on Domino a "
            "component on served is called with the run token)"
        )
    return said + _where_the_key_goes(name, keyed, entry, variable_of)


def _where_the_key_goes(
    name: str, keyed: str, entry: Mapping[str, Any], variable_of: VariableOf
) -> str:
    place = f"components.{name}.{keyed}.{_KEY}"
    named = _named_variable(entry.get(keyed))
    if named:
        return (
            f"the {name} component's key block, {place}, reads its API key from "
            f"the environment variable {named}: set {named} to the API key instead"
        )
    source = variable_of("components", name, keyed, _KEY, _SOURCE)
    return (
        f"the {name} component names where its API key comes from in a key block, "
        f"{place}, such as {_EXAMPLES} (by variables, {source} and the like); "
        "with no key block, no key is sent"
    )


def _named_variable(block: object) -> str:
    # The variable the block's key block names, when it is shaped as a name.
    key = block.get(_KEY) if isinstance(block, Mapping) else None
    named = key.get("variable") if isinstance(key, Mapping) else None
    return named if isinstance(named, str) and is_a_name(named) else ""


def _block_taking_a_key(
    name: str, entry: Mapping[str, Any], held: tuple[str, ...]
) -> str | None:
    # The block of the component's settings that takes a key block: the one
    # holding api_key when it does, else the first; None for a type with none.
    type_name = str(entry.get("type", name))
    if type_name not in COMPONENT_TYPE_SETTINGS.names():
        return None
    fields = COMPONENT_TYPE_SETTINGS.get(type_name).model_fields
    keyed = [block for block, field in fields.items() if _takes_a_key(field)]
    return held[0] if held and held[0] in keyed else next(iter(keyed), None)


def _takes_a_key(field: FieldInfo) -> bool:
    block = field.annotation
    return (
        isinstance(block, type)
        and issubclass(block, BaseModel)
        and (_KEY in block.model_fields)
    )


def file_key_blocks_given_above_left_out(
    file_values: dict[str, Any], above: Mapping[str, Any]
) -> dict[str, Any]:
    """Leave out of the workflow file's values each key block a place above gives.

    Args:
        file_values: The workflow file's values, as read.
        above: What the places above the file gave, merged (the command's
            values, the environment, ``.env``).

    Returns:
        ``file_values``, without the ``key`` of each block whose key block the
        places above give with a ``source``; unchanged when there is none.

    Example:
        >>> file_key_blocks_given_above_left_out(
        ...     {"components": {"llm": {"api": {"key": {"source": "env"}}}}},
        ...     {"components": {"llm": {"api": {"key": {"source": "domino"}}}}})
        {'components': {'llm': {'api': {}}}}
    """
    given = _key_blocks_with_a_source(above.get("components"))
    components = file_values.get("components")
    if not given or not isinstance(components, Mapping):
        return file_values
    kept = dict(components)
    for name, block in given:
        kept[name] = _without_key(kept.get(name), block)
    return {**file_values, "components": kept}


def _key_blocks_with_a_source(components: object) -> list[tuple[str, str]]:
    return [
        (name, block_name)
        for name, entry in _mappings(components)
        for block_name, block in _mappings(entry)
        if isinstance(block.get(_KEY), Mapping) and _SOURCE in block[_KEY]
    ]


def _without_key(entry: object, block_name: str) -> object:
    block = entry.get(block_name) if isinstance(entry, Mapping) else None
    if not isinstance(entry, Mapping) or not isinstance(block, Mapping):
        return entry
    return {**entry, block_name: {k: v for k, v in block.items() if k != _KEY}}


class KeyBlocksChecked(PydanticBaseSettingsSource):
    """The last settings source: refuses key mistakes in what the others gave.

    It reads the merged values of every source before it, gives none of its
    own, and raises before the values are validated, so a refusal is one line
    that quotes no value (``refuse_key_mistakes``).

    Args:
        settings_cls: The settings class being built.
        variable_of: Names the variable a value is read from.

    Example:
        >>> KeyBlocksChecked(WorkflowSettings, WorkflowSettings.variable_of)
    """

    def __init__(
        self, settings_cls: type[BaseSettings], variable_of: VariableOf
    ) -> None:
        super().__init__(settings_cls)
        self._variable_of = variable_of

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """Give no value for a field: this source only checks.

        Args:
            field: The field.
            field_name: Its name.

        Returns:
            None, the name, and not complex.
        """
        del field
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Check the components the sources before it gave; give nothing.

        Returns:
            An empty mapping.

        Raises:
            ValueError: The first key mistake (``refuse_key_mistakes``).
        """
        components = self.current_state.get("components")
        refuse_key_mistakes(components, self._variable_of)
        return {}
