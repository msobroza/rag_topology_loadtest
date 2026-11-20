"""The ``key`` block: where an endpoint's or the store's API key comes from.

The block names a key source and holds the keys that source reads, flat:
``{source: env, variable: OPENAI_API_KEY}`` or ``{source: domino}``. With no
block, no key is sent. The sources are registered in a package the settings
may not import, so the settings take any name and any keys beside ``source``,
and the start-up check of registered names refuses an unknown source and a key
its source does not read (``composition/registered_names.py``). The old
``api_key`` of an ``api`` block is refused here, from whichever of the workflow
file, the environment or ``.env`` it comes, in one line that names the block to
write and never the value, by the source that reads the workflow file. So is
the old ``api_key`` of the vector store's ``elasticsearch`` block, which takes
a ``key`` block too.

Example:
    >>> KeySettings(source="env", variable="OPENAI_API_KEY").read_keys()
    {'variable': 'OPENAI_API_KEY'}
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ConfigDict
from pydantic.fields import FieldInfo
from pydantic_settings import PydanticBaseSettingsSource

from .settings_blocks import SettingsBlock

# The key of a block that held the key itself before key sources.
_OLD_KEY = "api_key"
# The blocks that held it then, each holding a key block now: a closed list,
# what the settings were before key sources, which no registry gives.
_BLOCKS_WITH_AN_OLD_KEY = ("api", "elasticsearch")


class KeySettings(SettingsBlock):
    """A ``key`` block: a key source's name and the keys that source reads.

    Example:
        >>> KeySettings(source="domino").read_keys()
        {}
    """

    # The keys beside source are the source's own: checked at start-up against
    # what the source registered under that name reads.
    model_config = ConfigDict(frozen=True, extra="allow")

    source: str

    def read_keys(self) -> dict[str, str]:
        """Return the block's keys beside ``source``.

        Returns:
            Each key and its value, as the workflow file or a variable gave it.

        Example:
            >>> KeySettings(source="env", variable="X").read_keys()
            {'variable': 'X'}
        """
        return {key: str(value) for key, value in (self.model_extra or {}).items()}


class OldApiKeysRefused(PydanticBaseSettingsSource):
    """The workflow file's source, refusing the old ``api_key`` of a block.

    It is read after the constructor's values, the environment and ``.env``,
    so it holds what they gave and checks it with the file's own values. The
    refusal is raised here, before the values are validated, so it is one
    line: pydantic would quote every value given beside it, the key included.

    Args:
        workflow: The workflow file's source.
        variable_of: Names the variable a value of the workflow file is read
            from (``WorkflowSettings.variable_of``).

    Example:
        >>> OldApiKeysRefused(workflow_file_source(cls, path), cls.variable_of)()
        ValueError: QP_COMPONENTS__LLM__API__API_KEY (components.llm.api.api_key ...
    """

    def __init__(
        self, workflow: PydanticBaseSettingsSource, variable_of: Callable[..., str]
    ) -> None:
        super().__init__(workflow.settings_cls)
        self._workflow, self._variable_of = workflow, variable_of

    def _set_current_state(self, state: dict[str, Any]) -> None:
        super()._set_current_state(state)
        self._workflow._set_current_state(state)

    def _set_settings_sources_data(self, states: dict[str, dict[str, Any]]) -> None:
        super()._set_settings_sources_data(states)
        self._workflow._set_settings_sources_data(states)

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """Return what the workflow file's source gives for one field.

        Args:
            field: The field.
            field_name: Its name.

        Returns:
            The value, its key and whether it is complex, as the file's source
            returns them.
        """
        return self._workflow.get_field_value(field, field_name)

    def __call__(self) -> dict[str, Any]:
        """Return the workflow file's values, once no source holds an old key.

        Returns:
            The file's values.

        Raises:
            ValueError: The first component whose ``api`` or ``elasticsearch``
                block holds ``api_key``, naming its variable, its key in a
                workflow file, the component and the ``key`` block to write;
                never the value.
        """
        values = self._workflow()
        for read in (self.current_state, values):
            refuse_old_api_keys(read.get("components"), self._variable_of)
        return values


def refuse_old_api_keys(components: object, variable_of: Callable[..., str]) -> None:
    """Refuse a component whose ``api`` or ``elasticsearch`` block holds ``api_key``.

    The one writer of that line: start-up raises it from the workflow file's
    source, and the deploy tool for the ``components`` of its overrides.

    Args:
        components: A ``components`` block as read, any shape: a mapping of
            names to blocks is checked, anything else passes.
        variable_of: Names the variable a value of the workflow file is read
            from (``WorkflowSettings.variable_of``).

    Raises:
        ValueError: For the first such component: its old variable, its key in
            a workflow file, the ``key`` block to write; never the value.

    Example:
        >>> refuse_old_api_keys({"llm": {"api": {"api_key": "x"}}}, variable_of)
        ValueError: QP_COMPONENTS__LLM__API__API_KEY (components.llm.api.api_key ...
    """
    if not isinstance(components, Mapping):
        return
    for name, entry in components.items():
        held = _blocks_with_an_old_key(entry)
        if held:
            raise ValueError(_old_api_key_message(str(name), held[0], variable_of))


def _blocks_with_an_old_key(entry: object) -> list[str]:
    # The names of a component's blocks that still hold the old api_key.
    if not isinstance(entry, Mapping):
        return []
    held = [(name, entry.get(name)) for name in _BLOCKS_WITH_AN_OLD_KEY]
    return [
        name for name, block in held if isinstance(block, Mapping) and _OLD_KEY in block
    ]


def _old_api_key_message(name: str, block: str, variable_of: Callable[..., str]) -> str:
    old = variable_of("components", name, block, _OLD_KEY)
    source = variable_of("components", name, block, "key", "source")
    path = f"components.{name}.{block}"
    return (
        f"{old} ({path}.{_OLD_KEY} in a workflow file) is no longer read: the "
        f"{name} component names where its API key comes from in a key block, "
        f"{path}.key, such as {{source: env, variable: <the variable that holds "
        "the key>} or {source: domino} (by variables, "
        f"{source} and the like); with no key block, no key is sent"
    )
