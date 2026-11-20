"""A fixed API key, the value an endpoint's ``api_key`` setting gives, behind the port.

Leaves with ticket 02 of "an API key comes from a key source": the ``key`` block
then names the ``env`` source, which reads the key from a variable, and the
``api_key`` setting goes. Until then a fixed key takes the same road as the run
token, so its 401 runs out the retries of its ``http`` block. It is registered
under no name: no workflow file can name it.

Example:
    >>> FixedKey("sk-...").key(may_ask=False)
    'sk-...'
"""

from __future__ import annotations

from .base_key_source import BaseKeySource


class FixedKey(BaseKeySource):
    """A key read once from the settings, which a refusal cannot change.

    Args:
        value: The key, never empty (an empty ``api_key`` sends no key).

    Example:
        >>> FixedKey("sk-...").refused("sk-...")  # it keeps the key
    """

    def __init__(self, value: str) -> None:
        self._value = value

    def key(self, *, may_ask: bool) -> str:
        """Give the key: it is always held, so nothing is ever asked.

        Args:
            may_ask: Unused: a fixed key never asks.

        Returns:
            The key.

        Example:
            >>> FixedKey("sk-...").key(may_ask=True)
            'sk-...'
        """
        del may_ask
        return self._value

    def refused(self, key: str) -> None:
        """Keep the key: a fixed key cannot change while the process runs.

        Args:
            key: The key the refused call carried.

        Example:
            >>> FixedKey("sk-...").refused("sk-...")
        """
        del key
