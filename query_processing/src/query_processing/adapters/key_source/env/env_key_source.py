"""The ``env`` key source: an API key an environment variable holds.

The block names the variable (``{source: env, variable: OPENAI_API_KEY}``). It is
read where the settings are read: the process's environment first, then the
``.env`` file the settings read, so a laptop run needs no ``export``. It is read
once, when the source opens, and held for the life of the process: a process
does not see its environment change, so a refusal changes nothing, and the
error of a refused call says to restart the App after changing the variable.

Example:
    >>> source = EnvKeySource("OPENAI_API_KEY", env_file=Path(".env"))
    >>> source.key(may_ask=False)
    'sk-...'
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from dotenv import dotenv_values

from ....domain.errors import MissingKeyError
from ..base_key_source import KEY_SOURCES, BaseKeySource


class EnvKeySource(BaseKeySource):
    """A key read from a variable when the source opens, held until the process ends.

    Args:
        variable: The environment variable that holds the key.
        env_file: The ``.env`` file read when the environment lacks the
            variable; None or a missing file for none.

    Raises:
        MissingKeyError: If the variable is unset or empty in both, naming it.

    Example:
        >>> EnvKeySource("OPENAI_API_KEY", env_file=None).described()
        'its key is read from the environment variable OPENAI_API_KEY ...'
    """

    reads: ClassVar[tuple[str, ...]] = ("variable",)

    def __init__(self, variable: str, *, env_file: Path | None) -> None:
        key = os.environ.get(variable) or _from_dotenv(variable, env_file)
        self._variable, self._key = variable, key
        if not key:
            where = f" and in {env_file}" if env_file is not None else ""
            raise MissingKeyError(
                f"the environment variable {variable} is unset or empty, in the "
                f"environment{where} (expected the API key)"
            )

    @classmethod
    def from_block(
        cls, keys: Mapping[str, str], *, env_file: Path | None
    ) -> EnvKeySource:
        """Open the source of a block ``{source: env, variable: X}``.

        Args:
            keys: The block's ``variable``.
            env_file: The ``.env`` file the settings read.

        Returns:
            The source, its key read.

        Raises:
            MissingKeyError: If the variable is unset or empty.

        Example:
            >>> EnvKeySource.from_block({"variable": "X"}, env_file=None)
        """
        return cls(keys["variable"], env_file=env_file)

    def key(self, *, may_ask: bool) -> str:
        """Give the key read when the source opened: it never asks again.

        Args:
            may_ask: Unused: the key is always held.

        Returns:
            The key.

        Example:
            >>> source.key(may_ask=False)
            'sk-...'
        """
        del may_ask
        return self._key

    def refused(self, key: str) -> None:
        """Keep the key: the variable cannot change while the process runs.

        Args:
            key: The key the refused call carried.

        Example:
            >>> source.refused("sk-...")
        """
        del key

    def described(self) -> str:
        """Name the variable, and the restart a changed key needs.

        Returns:
            The clause an error of a refused call quotes.

        Example:
            >>> EnvKeySource("OPENAI_API_KEY", env_file=None).described()
            'its key is read from the environment variable OPENAI_API_KEY ...'
        """
        return (
            f"its key is read from the environment variable {self._variable} when "
            "the App starts: restart the App after changing it"
        )


def _from_dotenv(variable: str, env_file: Path | None) -> str:
    # The same file the settings read, parsed by the library they use.
    if env_file is None or not env_file.is_file():
        return ""
    return dotenv_values(env_file).get(variable) or ""


# Registered by call, as the domino source is: the decorator's return type
# would hide its constructor from the type checkers.
KEY_SOURCES.register("env")(EnvKeySource)
