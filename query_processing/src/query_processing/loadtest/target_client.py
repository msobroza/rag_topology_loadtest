"""The one HTTP call ``qp-record-answers`` makes: a JSON POST to the target, retried.

A request the target never answered (no connection, no answer within the
timeout) is sent again, ``retries`` times at most, after a wait of
``backoff_s`` that doubles before each next retry; then the call fails with
``TargetUnreachableError``. A response of any status is the target answering,
and is returned as it is: the record holds its status. The timeout, the
retries, the backoff and the headers come from the load test's settings; the
wait is injected, so a test does not wait.

Example:
    >>> client = httpx.Client(timeout=policy.timeout_s)
    >>> poster = TargetClient(client, "http://host:8888", {}, policy)
    >>> poster.post_json("/retrieve", {"question": "q"}).status_code
    200
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class RetryPolicy:
    """How long one attempt waits, how many more are made, the first wait between.

    Example:
        >>> RetryPolicy(timeout_s=60.0, retries=2, backoff_s=0.2).wait_before(2)
        0.4
    """

    timeout_s: float
    retries: int  # extra attempts after the first
    backoff_s: float  # before the first retry; doubles each time

    def wait_before(self, retry: int) -> float:
        """Return the seconds to wait before a retry.

        Args:
            retry: Which retry, 1 for the first.

        Returns:
            ``backoff_s``, doubled for each retry after the first.
        """
        return self.backoff_s * 2 ** (retry - 1)


class TargetUnreachableError(RuntimeError):
    """The target answered none of a request's attempts; the message says where.

    Example:
        >>> raise TargetUnreachableError("http://host/retrieve: no answer ...")
    """


class TargetClient:
    """Posts JSON bodies to one target under one retry policy.

    Args:
        client: What does the HTTP, with its own timeout; a test hands it an
            app in process.
        host: The target's base URL; a trailing slash is dropped.
        headers: Sent with every request: the load test's credentials.
        policy: The retries and the backoff between them.
        sleep: The wait before a retry.

    Example:
        >>> TargetClient(client, "http://host:8888", {}, policy).post_json("/query", {})
    """

    def __init__(
        self,
        client: httpx.Client,
        host: str,
        headers: Mapping[str, str],
        policy: RetryPolicy,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client, self._host = client, host.rstrip("/")
        self._headers, self._policy, self._sleep = dict(headers), policy, sleep

    def post_json(self, path: str, body: Mapping[str, object]) -> httpx.Response:
        """POST ``body`` to ``path`` on the target and return its response.

        Args:
            path: The route, such as ``/retrieve``.
            body: The JSON body.

        Returns:
            The first response the target gave, of any status.

        Raises:
            TargetUnreachableError: If no attempt got an answer; the message
                names the URL, the attempts and the last error.
        """
        url = f"{self._host}{path}"
        error: httpx.TransportError | None = None
        for attempt in range(self._policy.retries + 1):
            if attempt:
                self._sleep(self._policy.wait_before(attempt))
            try:
                return self._send(url, body)
            except httpx.TransportError as exc:
                error = exc
        raise TargetUnreachableError(
            f"{url}: no answer after {self._policy.retries + 1} attempt(s) of "
            f"{self._policy.timeout_s} s each; last error: {error!r}"
        )

    def _send(self, url: str, body: Mapping[str, object]) -> httpx.Response:
        # The timeout is the client's own: the command opens it with the
        # policy's, and a client handed in by a test answers in process.
        return self._client.post(url, json=dict(body), headers=self._headers)
