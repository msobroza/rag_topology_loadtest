"""The ``Authorization`` header of a model endpoint, from the key source it is given.

The key source is a port among the adapters, which this layer may not import:
``KeySupplier`` states the three methods it is called through. A key the source
already holds is taken on the event loop; a source that must ask is called off
it, in a thread, since its methods block (``key_of``, which the Elasticsearch
store takes its key with too). The key a refused call carried is
read back from that call's own headers, so a refusal names the key it was for
even with calls in flight that carry another.

Example:
    >>> bearer = BearerKey(source)
    >>> await bearer.headers()
    {'Authorization': 'Bearer <key>'}
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Protocol

_BEARER = "Bearer "


class KeySupplier(Protocol):
    """What the endpoint client calls of a key source (``BaseKeySource``).

    Example:
        >>> supplier.key(may_ask=False)
        'run-token'
    """

    def key(self, *, may_ask: bool) -> str | None:
        """Give the key held, asking for one only when it must and may.

        Args:
            may_ask: Whether it may ask its source, a call that may block.

        Returns:
            The key; None only when it would have to ask and may not.
        """
        ...

    def refused(self, key: str) -> None:
        """Be told that a call carrying ``key`` was refused for it.

        Args:
            key: The key the refused call carried.
        """
        ...

    def described(self) -> str:
        """Say where the key comes from and how it is changed, without the key.

        Returns:
            One clause, for the error of a call refused for its key.
        """
        ...


async def key_of(source: KeySupplier) -> str | None:
    """Take the key a source holds on the event loop, and ask for one off it.

    The one place code that awaits takes a key from a source, the endpoint
    client's ``BearerKey`` and the Elasticsearch store alike.

    Args:
        source: Where the key comes from.

    Returns:
        The key; "" or None when the source gives none.

    Raises:
        RagDependencyError: If the source could not get its key.

    Example:
        >>> await key_of(source)
        'run-token'
    """
    key = source.key(may_ask=False)
    if key is None:
        key = await asyncio.to_thread(source.key, may_ask=True)
    return key


class BearerKey:
    """Builds every call's ``Authorization`` header from one key source.

    Args:
        source: Where the key comes from.

    Example:
        >>> await BearerKey(source).headers()
        {'Authorization': 'Bearer abc'}
    """

    def __init__(self, source: KeySupplier) -> None:
        self._source = source

    async def headers(self) -> dict[str, str]:
        """Return the headers of one attempt, leaving the loop only to ask.

        Returns:
            ``{"Authorization": "Bearer <key>"}``, or ``{}`` if the source gave
            no key even when it could ask.

        Raises:
            RagDependencyError: If the source could not get its key.
        """
        key = await key_of(self._source)
        return {"Authorization": f"{_BEARER}{key}"} if key else {}

    def refused_in(self, sent: Mapping[str, str]) -> None:
        """Tell the source that the key of a refused call's headers was refused.

        Args:
            sent: The headers the refused call carried, as ``headers`` built them.

        Example:
            >>> bearer.refused_in({"Authorization": "Bearer abc"})
        """
        authorization = sent.get("Authorization", "")
        if authorization.startswith(_BEARER):
            self._source.refused(authorization.removeprefix(_BEARER))

    def described(self) -> str:
        """Say where the key comes from, for the error of a refused call.

        Returns:
            What the key source says of itself, never the key.

        Example:
            >>> bearer.described()
            'its key is read from the environment variable OPENAI_API_KEY ...'
        """
        return self._source.described()
