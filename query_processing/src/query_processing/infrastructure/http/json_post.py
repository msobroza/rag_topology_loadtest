"""POST a JSON body under one :class:`HttpPolicy`, retrying what a retry may fix.

Each attempt times out after ``timeout_s``; transport errors, 5xx and the
caller's extra statuses are retried ``retries`` times, waiting
``policy.backoff_before(attempt)`` in between; any other 4xx fails at once. The
model endpoints (``endpoint/api_endpoint_client.py``) post through it; they
differ only in their headers and extra retryable statuses. The headers a
caller supplies carry its key alone, so the body a failure quotes has every
credential they sent masked: an endpoint that echoes the key it was sent does
not put it in our error. ``keys_masked`` does the same for any other error
that quotes what was sent, such as the Elasticsearch client's.

Example:
    >>> rate_limited = frozenset({HTTPStatus.TOO_MANY_REQUESTS})
    >>> post = RetryingJsonPost(client, policy, headers=token.headers,
    ...                         retry_statuses=rate_limited)
    >>> body = await post.post("reranker", url, {"query": "q"})
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from http import HTTPStatus
from typing import NamedTuple

import httpx
from pydantic import ValidationError

from ...domain.errors import RagDependencyError, describe_error
from .http_policy import HttpPolicy

HeaderSupplier = Callable[[], Awaitable[dict[str, str]]]
# Called with a retried status and the headers of the attempt that got it.
StatusHook = Callable[[int, Mapping[str, str]], None]
# Called with a failed status; what it returns is added to the failure's detail.
StatusNote = Callable[[int], str]
# What stands in a quoted body for a credential the request sent.
_MASK = "***"


def _ignore_status(status: int, sent: Mapping[str, str]) -> None:
    # Default hook: most callers have nothing to reset before a retry.
    del status, sent


def _no_note(status: int) -> str:
    # Default note: most callers have nothing to add to a status.
    del status
    return ""


def keys_masked(text: str, keys: Iterable[str]) -> str:
    """Return ``text`` with every one of ``keys`` in it replaced by ``***``.

    Args:
        text: What an error quotes, such as a body or a client's exception.
        keys: The keys a call sent; an empty one masks nothing.

    Returns:
        The text, holding none of the keys.

    Example:
        >>> keys_masked("key sk-1 refused", ["sk-1"])
        'key *** refused'
    """
    for key in keys:
        if key:
            text = text.replace(key, _MASK)
    return text


def _masked(text: str, sent: Mapping[str, str]) -> str:
    # Each header sent is a credential, "Bearer <key>": its last word is the key.
    return keys_masked(text, (value.rsplit(" ", 1)[-1] for value in sent.values()))


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
        on_retry_status: Called with each retryable status and the headers of
            the attempt that got it, before the retry, so a caller can tell a
            key source which key was refused.
        note_on_status: Called with a failed attempt's status; what it returns
            ends that attempt's detail, so a caller can say where a refused key
            comes from.

    Example:
        >>> post = RetryingJsonPost(client, policy, headers=token.headers,
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
        note_on_status: StatusNote = _no_note,
    ) -> None:
        self._client, self._policy, self._headers = client, policy, headers
        self._retry_statuses, self._on_retry_status = retry_statuses, on_retry_status
        self._note_on_status = note_on_status

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
        # Masked before the cut, which could leave a key's first characters.
        quoted = _masked(response.text, headers)[:200]
        detail = f"HTTP {status}: {quoted}{self._note_on_status(status)}"
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR or status in self._retry_statuses:
            self._on_retry_status(status, headers)
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
