"""The ``key`` block: where an endpoint's or the store's API key comes from.

The block names a key source and holds the keys that source reads, flat:
``{source: env, variable: OPENAI_API_KEY}`` or ``{source: domino}``. With no
block, no key is sent. The sources are registered in a package the settings
may not import, so the settings take any name and any keys beside ``source``,
and the start-up check of registered names refuses an unknown source and a key
its source does not read (``composition/registered_names.py``). Before the
settings library reads a block, ``key_blocks.py`` refuses one that is not
shaped as a block, and the old ``api_key``, in lines that never quote a value.

Example:
    >>> KeySettings(source="env", variable="OPENAI_API_KEY").read_keys()
    {'variable': 'OPENAI_API_KEY'}
"""

from __future__ import annotations

from pydantic import ConfigDict

from .settings_blocks import SettingsBlock


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
