"""The Elasticsearch store's API key: the client opened with it, and again once refused.

Elasticsearch's client takes its key when it is built, as an Elasticsearch API
key (``ApiKey <key>``), not a bearer header. So a key its source rotated is
taken by building a new client: told of Elasticsearch's 401, this tells the
source which key was refused and builds the client again with the key the
source then gives. A source that still holds the refused key, such as ``env``
until its App restarts, gets no new client: the call fails at once with what
the source says of itself. The store closes the client it replaced only when it
closes itself, so a call in flight on it ends. Without a source the client is
built with no key, and basic authentication written in the URL is the
client's own. Every key sent is masked in an error.

Example:
    >>> key = ElasticsearchClientKey(connect, source)
    >>> langchain_store = await key.connected()
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ....domain.errors import describe_error
from ....infrastructure.endpoint.bearer_key import key_of
from ....infrastructure.http.json_post import keys_masked
from ...key_source.base_key_source import BaseKeySource

# Builds LangChain's store from a key, or with none: a test gives a fake one.
StoreConnector = Callable[[str | None], Any]


class ElasticsearchClientKey:
    """Builds the store's LangChain store with its source's key, and again once refused.

    Args:
        connect: Builds LangChain's store, its client sending the key given.
        source: Where the key comes from; None builds it with no key.

    Example:
        >>> key = ElasticsearchClientKey(connect, None)
        >>> langchain_store = await key.connected()  # no key sent
    """

    def __init__(self, connect: StoreConnector, source: BaseKeySource | None) -> None:
        self._connect, self._source = connect, source
        self._sent: list[str] = []  # each key a client was built with, the last in use

    async def connected(self) -> Any:
        """Build LangChain's store with the key the source gives now.

        Returns:
            The new store; the caller keeps the one it replaces until it closes.

        Raises:
            RagDependencyError: If the source could not get its key.
        """
        key = await key_of(self._source) if self._source is not None else None
        if key:
            self._sent.append(key)
        return self._connect(key or None)

    async def reconnected(self) -> Any | None:
        """Tell the source its key was refused and build the store with its next one.

        Returns:
            The new store, which the caller keeps the replaced one beside until
            it closes; None when the source still holds the refused key, such
            as an ``env`` source, so that a new client would be refused too.

        Raises:
            RagDependencyError: If the source could not get its key.
        """
        if self._source is None or not self._sent:
            return await self.connected()
        refused = self._sent[-1]
        self._source.refused(refused)
        if self._source.key(may_ask=False) == refused:
            return None
        return await self.connected()

    def rotates_on(self, failure: BaseException) -> bool:
        """Say whether a failure is a refusal of a key its source may change.

        Args:
            failure: What a call of the client raised.

        Returns:
            True for Elasticsearch's 401 with a source; a 403 is not a refused key.
        """
        # Imported here, as the client is in ``open``: importing the store stays cheap.
        from elasticsearch import AuthenticationException

        refused = isinstance(failure, AuthenticationException)
        return refused and self._source is not None

    def described(self, failure: BaseException) -> str:
        """Describe a failure in one line, no key in it, and a refused key's source.

        Args:
            failure: What a call of the client raised.

        Returns:
            The failure's type and text with every key sent masked, and, for a
            refused key, what its source says of itself.
        """
        detail = keys_masked(describe_error(failure), self._sent)
        if self._source is None or not self.rotates_on(failure):
            return detail
        return f"{detail} (the key was refused: {self._source.described()})"
