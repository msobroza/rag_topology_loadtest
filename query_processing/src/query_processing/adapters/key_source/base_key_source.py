"""The key source port: where an API key comes from, and what it is told on a refusal.

A key source gives the API key it holds and asks its source only when it holds
none it may still give. Told that a key was refused (an HTTP 401), it forgets
that key if it still holds it, so the next call asks again. Its methods do not
await: code that awaits asks for a held key on the event loop and calls the
source off it only when the source must ask. Each source is one directory
under this package, registered in ``KEY_SOURCES`` by its name. This module
imports the domain only.

Example:
    >>> source = KEY_SOURCES.get("domino")(client, token_url=url, ttl_s=240.0,
    ...                                    policy=policy)
    >>> source.key(may_ask=True)
    'eyJ...'
"""

from __future__ import annotations

import abc

from ...domain.registry import Registry


class BaseKeySource(abc.ABC):
    """Where one API key comes from: given when held, asked for when not.

    Example:
        >>> held = source.key(may_ask=False)  # never asks: None when it must
        >>> key = held if held is not None else source.key(may_ask=True)
        >>> source.refused(key)  # the call carrying it got a 401
    """

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


# The key sources by the name a setting gives; each directory registers its own.
KEY_SOURCES: Registry[type[BaseKeySource]] = Registry("key source")
