"""What every port factory does: the port supplied, else its Backend's, opened once.

A factory builds the implementation of one port from the settings. Its subclass
says which setting holds the Backend and maps each Backend value to the method
that opens it, so a new Backend is one method and one entry, in that factory
only. A port handed in ready-made (a test double, mostly) is used as given and
never closed here.

Example:
    >>> class LLMFactory(BasePortFactory[BaseLLM]):
    ...     port = "llm"
    ...     def openers(self): return {"api": self._api, "fake": self._fake}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar, Generic, TypeVar

from ..config.rag_settings import RagSettings
from .resource_scope import Opener, ResourceScope

T = TypeVar("T")


class BasePortFactory(ABC, Generic[T]):
    """Opens one port: the one supplied, else the implementation of its Backend.

    Args:
        settings: Where the Backend and every value an implementation takes are
            read.
        scope: Opens the port once and reuses it until the scope ends.
        supplied: A ready-made port, used as given; None to build one.

    Example:
        >>> embedder = await EmbedderFactory(settings, scope, http, models).open()
    """

    port: ClassVar[str]  # names the port in the scope and in the errors

    def __init__(
        self, settings: RagSettings, scope: ResourceScope, supplied: T | None
    ) -> None:
        self._settings, self._scope, self._supplied = settings, scope, supplied

    @abstractmethod
    def backend(self) -> str:
        """Return the Backend the settings choose for this port.

        Returns:
            One of the keys of ``openers``.
        """

    @abstractmethod
    def openers(self) -> Mapping[str, Opener[T]]:
        """Return every Backend of this port and what opens its implementation.

        Returns:
            One entry per Backend value.
        """

    async def open(self) -> T:
        """Open the port: the supplied one, else the one its Backend names.

        Returns:
            The port, the same object on every call within the scope.

        Example:
            >>> reranker = await reranker_factory.open()
        """
        if self._supplied is not None:
            return self._supplied
        return await self._scope.once(self.port, self.backend(), self._open_backend)

    async def _open_backend(self) -> T:
        return await self.openers()[self.backend()]()
