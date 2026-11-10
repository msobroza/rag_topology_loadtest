"""Errors shared by every layer: a dependency failure and its one-line description."""

from __future__ import annotations

# --- errors ----------------------------------------------------------------


def describe_error(exc: BaseException) -> str:
    """The exception's type, plus its message when it has one.

    Example::

        describe_error(TimeoutError("read timed out"))  # "TimeoutError: read timed out"
    """
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class RagDependencyError(RuntimeError):
    """A dependency failed; ``kind`` is ``unavailable`` (HTTP 503) or ``timeout`` (504).

    Example::

        raise RagDependencyError("reranker", "HTTP 503", target=url)
    """

    def __init__(
        self,
        component: str,
        detail: str,
        *,
        target: str | None = None,
        kind: str = "unavailable",
    ) -> None:
        self.component, self.detail, self.target, self.kind = (
            component,
            detail,
            target,
            kind,
        )
        suffix = f" (target={target})" if target else ""
        super().__init__(f"{component} {kind}: {detail}{suffix}")
