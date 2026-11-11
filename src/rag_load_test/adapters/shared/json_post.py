"""POST a JSON body under one :class:`HttpPolicy`, retrying what a retry may fix.

Each attempt times out after ``timeout_s``; transport errors, 5xx and the
caller's extra statuses are retried ``retries`` times, waiting
``policy.backoff_before(attempt)`` in between; any other 4xx fails at once. The
OVMS adapters and the hosted rerank route share it; they differ only in their
headers and extra retryable statuses.

Example:
    >>> rate_limited = frozenset({HTTPStatus.TOO_MANY_REQUESTS})
    >>> post = RetryingJsonPost(client, policy, headers=no_headers,
    ...                         retry_statuses=rate_limited)
    >>> body = await post.post("reranker", url, {"query": "q"})
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from typing import NamedTuple

import httpx
from pydantic import ValidationError

from ...domain.errors import RagDependencyError, describe_error
from .http_policy import HttpPolicy

HeaderSupplier = Callable[[], Awaitable[dict[str, str]]]
StatusHook = Callable[[int], None]


async def no_headers() -> dict[str, str]:
    """Supply no headers, for a route that needs no auth.

    Returns:
        An empty dict.

    Example:
        >>> await no_headers()
        {}
    """
    return {}


def _ignore_status(status: int) -> None:
    # Default hook: most callers have nothing to reset before a retry.
    del status


class _RetryableFailure(NamedTuple):
    # An attempt that failed in a way a retry may fix, and how to report it.
    detail: str
    kind: str


class RetryingJsonPost:
    """POST JSON and return the decoded body, retrying per one policy.

    Args:
        client: The shared HTTP client every call goes through.
        policy: Timeout, retries and backoff of every call.
        headers: Supplies the headers of each attempt, called once per attempt.
        retry_statuses: 4xx statuses retried like a 5xx (e.g. 429).
        on_retry_status: Called with each retryable status before the retry, so
            a caller can drop a stale token.

    Example:
        >>> post = RetryingJsonPost(client, policy, headers=no_headers,
        ...                         retry_statuses=frozenset())
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        policy: HttpPolicy,
        *,
        headers: HeaderSupplier,
        retry_statuses: frozenset[int],
        on_retry_status: StatusHook = _ignore_status,
    ) -> None:
        self._client, self._policy, self._headers = client, policy, headers
        self._retry_statuses, self._on_retry_status = retry_statuses, on_retry_status

    async def post(self, component: str, url: str, payload: object) -> object:
        """POST ``payload`` until an attempt succeeds or the retries run out.

        Args:
            component: Names the dependency in any error, such as ``reranker``.
            url: Where to post.
            payload: The JSON body.

        Returns:
            The decoded JSON body of the first answer below 400.

        Raises:
            RagDependencyError: When every attempt failed in a way a retry may
                fix; the error carries the last attempt's detail and kind.
        """
        failure = _RetryableFailure("no attempt made", "unavailable")
        for attempt in range(self._policy.retries + 1):
            if attempt:
                await asyncio.sleep(self._policy.backoff_before(attempt))
            outcome = await self._attempt(component, url, payload)
            if not isinstance(outcome, _RetryableFailure):
                return outcome
            failure = outcome
        raise RagDependencyError(
            component, failure.detail, target=url, kind=failure.kind
        )

    async def _attempt(self, component: str, url: str, payload: object) -> object:
        # One POST, sorted into a body, a _RetryableFailure, or a raised
        # RagDependencyError for a 4xx a retry cannot fix.
        try:
            headers = await self._headers()
            response = await self._client.post(
                url, json=payload, headers=headers, timeout=self._policy.timeout_s
            )
        except httpx.TransportError as exc:
            timed_out = isinstance(exc, httpx.TimeoutException)
            kind = "timeout" if timed_out else "unavailable"
            return _RetryableFailure(describe_error(exc), kind)
        status = response.status_code
        if status < HTTPStatus.BAD_REQUEST:
            return response.json()
        detail = f"HTTP {status}: {response.text[:200]}"
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR or status in self._retry_statuses:
            self._on_retry_status(status)
            return _RetryableFailure(detail, "unavailable")
        raise RagDependencyError(component, detail, target=url)


def describe_malformed(exc: Exception) -> str:
    """Name what broke a response's shape in one line, not the whole (maybe huge) body.

    Args:
        exc: The error reading the body: a pydantic ``ValidationError`` or any
            other exception.

    Returns:
        ``ValidationError for <model>: <field>: <msg>; ...`` for a validation
        error, else the exception's one-line description.

    Example:
        >>> describe_malformed(IndexError("rerank result index 5 is outside ..."))
        'IndexError: rerank result index 5 is outside ...'
    """
    if not isinstance(exc, ValidationError):
        return describe_error(exc)
    problems = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<body>'}: {error['msg']}"
        for error in exc.errors()
    )
    return f"ValidationError for {exc.title}: {problems}"
