"""The key source port: where an API key comes from, and what it is told on a refusal.

A key source gives the API key it holds and asks its source only when it holds
none it may still give. Told that a key was refused (an HTTP 401), it forgets
that key if it still holds it, so the next call asks again. It says, for an
error, where its key comes from and how the key is changed, never the key. Its
methods do not await: code that awaits asks for a held key on the event loop
and calls the source off it only when the source must ask. Each source is one
directory under this package, registered in ``KEY_SOURCES`` by the name a
``key`` block's ``source`` gives, and declares in ``reads`` the other keys of
that block, which the start-up check of registered names holds a block to.
This module imports the domain only.

Example:
    >>> source = KEY_SOURCES.get("env").from_block(
    ...     {"variable": "OPENAI_API_KEY"}, env_file=Path(".env"))
    >>> source.key(may_ask=True)
    'sk-...'
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from ...domain.registry import Registry


class BaseKeySource(abc.ABC):
    """Where one API key comes from: given when held, asked for when not.

    Example:
        >>> held = source.key(may_ask=False)  # never asks: None when it must
        >>> key = held if held is not None else source.key(may_ask=True)
        >>> source.refused(key)  # the call carrying it got a 401
    """

    # The keys of its ``key`` block beside ``source``, each one required: the
    # one declaration of what a block of this source may hold.
    reads: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_block(
        cls, keys: Mapping[str, str], *, env_file: Path | None
    ) -> BaseKeySource:
        """Open the source a ``key`` block names, from the keys it ``reads``.

        A source whose constructor takes those keys alone keeps this; one that
        needs more overrides it. The ``domino`` source is the process's one and
        is opened by the composition root, not from a block.

        Args:
            keys: The block's keys beside ``source``, as ``reads`` names them.
            env_file: The ``.env`` file the settings read, or None.

        Returns:
            The opened source.

        Raises:
            MissingKeyError: If the key it must give is nowhere to be read.

        Example:
            >>> KEY_SOURCES.get("env").from_block({"variable": "X"},
            ...                                   env_file=None).described()
        """
        del env_file
        return cls(**keys)

    @abc.abstractmethod
    def key(self, *, may_ask: bool) -> str | None:
        """Give the key it holds, asking its source first when it must and may.

        Args:
            may_ask: Whether it may ask its source (a call that may block)
                when it holds no key it may still give.

        Returns:
            The key. None only when ``may_ask`` is false and it would have to
            ask: the caller then asks again with ``may_ask``, off the loop.

        Raises:
            RagDependencyError: If asking its source failed, once the retries
                of the shared ``http`` block ran out.

        Example:
            >>> source.key(may_ask=False)
            'run-token'
        """

    @abc.abstractmethod
    def refused(self, key: str) -> None:
        """Be told that a call carrying ``key`` was refused for it (an HTTP 401).

        Args:
            key: The key the refused call carried. It is forgotten only if it
                is still the one held: a late refusal of an older key keeps the
                newer one.

        Example:
            >>> source.refused("run-token")
        """

    @abc.abstractmethod
    def described(self) -> str:
        """Say where its key comes from and how it is changed, without the key.

        Returns:
            One clause, which the error of a call refused for its key quotes.

        Example:
            >>> source.described()
            'its key is read from the environment variable OPENAI_API_KEY ...'
        """


# The key sources by the name a setting gives; each directory registers its own.
KEY_SOURCES: Registry[type[BaseKeySource]] = Registry("key source")
