"""A start-up refused for a settings mistake reaches the server as one line.

``qp-serve``'s app adds this middleware for a refused graph
(``entrypoints/qp_serve.py``); it lives here, out of that module, for the
500-line limit of a file.

Example:
    >>> app.add_middleware(
    ...     StartupRefusalInOneLine, refused=WorkflowGraphError, prog="qp-serve"
    ... )
"""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class StartupRefusalInOneLine:
    """ASGI middleware: a start-up refused with ``refused`` is logged as one line.

    Starlette hands the server a failed start-up as the whole traceback, which
    the server logs. A refusal once the components are open (a graph whose
    store ranks no text under a lexical retrieval) is a mistake in the
    settings, not a crash, so the server gets its message alone. Any other
    failure keeps its traceback.

    Args:
        app: The app whose start-up this watches.
        refused: The exception, or the exceptions, a settings mistake raises
            at start-up.
        prog: The command's name, which begins the line.

    Example:
        >>> app.add_middleware(
        ...     StartupRefusalInOneLine, refused=WorkflowGraphError, prog="qp-serve"
        ... )
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        refused: type[Exception] | tuple[type[Exception], ...],
        prog: str,
    ) -> None:
        self.app, self._refused, self._prog = app, refused, prog

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass a request on; hold a failed start-up until its cause is known.

        Args:
            scope: The ASGI scope; only a ``lifespan`` one is watched.
            receive: The server's receive channel.
            send: The server's send channel.

        Raises:
            Exception: A ``refused`` exception raised at start-up, re-raised
                once the server has its one-line message.
        """
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return
        # Held until the error that caused it is known, then sent once.
        failed: list[Message] = []

        async def hold_failure(message: Message) -> None:
            if message["type"] == "lifespan.startup.failed":
                failed.append(message)
            else:
                await send(message)

        try:
            await self.app(scope, receive, hold_failure)
        except self._refused as exc:
            if failed:
                one_line = f"{self._prog}: cannot start: {exc}".replace("\n", " ")
                failed[:] = [{**failed[0], "message": one_line}]
            raise
        finally:
            for message in failed:
                await send(message)
