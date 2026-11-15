"""What one process has opened: each thing once, all of it closed together.

Inside ``async with`` a named resource is opened on its first request and reused
until the block ends, when everything registered on the exit stack is closed, in
reverse order. A failure to open is reported as a ``RagDependencyError`` naming
the resource. First requests are not guarded against running concurrently: open
resources from one task (an app lifespan or a CLI ``main``).

Example:
    >>> async with ResourceScope("CompositionRoot") as scope:
    ...     client = await scope.once("http_client", OWNED, start_client)
    ...     scope.exit_stack("http_client").push_async_callback(client.aclose)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from types import TracebackType
from typing import TypeVar, cast

from ..domain.errors import RagDependencyError, describe_error

T = TypeVar("T")
Opener = Callable[[], Awaitable[T]]  # opens one resource
OWNED = "in-process"  # the "backend" of a resource the process creates for itself


class ResourceScope:
    """Opens each named resource once and closes them all when the block ends.

    Args:
        owner: Whose scope this is; it names the scope in the errors.

    Example:
        >>> async with ResourceScope("CompositionRoot") as scope:
        ...     pool = await scope.once("model_pool", OWNED, start_pool)
    """

    def __init__(self, owner: str) -> None:
        self._owner = owner
        self._opened: dict[str, object] = {}
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> ResourceScope:
        if self._exit_stack is not None:
            raise RuntimeError(
                f"{self._owner} is already open; exit it before re-entering"
            )
        self._exit_stack = AsyncExitStack()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        stack, self._exit_stack = self.exit_stack("__aexit__"), None
        self._opened.clear()
        await stack.aclose()

    async def once(self, name: str, backend: str, open_it: Opener[T]) -> T:
        """Open ``name`` on its first request and reuse it until the block ends.

        Args:
            name: The resource; one name is opened once.
            backend: What it is opened as; named in the error when it cannot open.
            open_it: Opens the resource; called on the first request only.

        Returns:
            The opened resource.

        Raises:
            RagDependencyError: If ``open_it`` fails; an error that already is one
                propagates as it is, since it names what failed further down.

        Example:
            >>> await scope.once("llm", "api", open_llm)
        """
        self.exit_stack(name)
        if name not in self._opened:
            try:
                self._opened[name] = await open_it()
            except RagDependencyError:
                raise  # a resource this one depends on already says what failed
            except Exception as exc:
                detail = f"could not open backend {backend!r}: {describe_error(exc)}"
                raise RagDependencyError(name, detail) from exc
        return cast(T, self._opened[name])

    def exit_stack(self, requested: str) -> AsyncExitStack:
        """Return the stack whose callbacks run when the block ends.

        Args:
            requested: What asks for it; named in the error.

        Returns:
            The exit stack of the open block.

        Raises:
            RuntimeError: If the scope is not open.

        Example:
            >>> scope.exit_stack("vector_store").push_async_callback(store.aclose)
        """
        if self._exit_stack is None:
            owner = self._owner
            raise RuntimeError(
                f"{owner}: {requested!r} requested outside 'async with {owner}(...)'"
            )
        return self._exit_stack
