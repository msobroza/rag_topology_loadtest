"""Errors shared by every layer: a failed dependency, its description, a missing key."""

from __future__ import annotations

# --- errors ----------------------------------------------------------------


def describe_error(exc: BaseException) -> str:
    """Describe an exception in one line.

    Args:
        exc: The exception to describe.

    Returns:
        The exception's type name, plus ``: <message>`` when it has a message.

    Example:
        >>> describe_error(TimeoutError("read timed out"))
        'TimeoutError: read timed out'
    """
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class RagDependencyError(RuntimeError):
    """A dependency failed.

    The message reads ``<component> <kind>: <detail> (target=<target>)``.

    Args:
        component: The dependency that failed, such as ``reranker``.
        detail: What went wrong, such as ``HTTP 503``.
        target: Where the dependency was called; added to the message when set.
        kind: ``unavailable`` (HTTP 503) or ``timeout`` (504).

    Example:
        >>> raise RagDependencyError("reranker", "HTTP 503", target=url)
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


class MissingKeyError(ValueError):
    """An API key a key source must give is nowhere to be read: a settings mistake.

    Raised when the source opens, such as an ``env`` source whose variable is
    unset or empty, so a command that opens no endpoint needs no key. It names
    where the key was looked for and never holds a key. The resource scope lets
    it through unwrapped, and ``qp-serve`` reports it as its start-up's one line.

    Example:
        >>> raise MissingKeyError("the 'llm' component's API key: X is unset")
    """
