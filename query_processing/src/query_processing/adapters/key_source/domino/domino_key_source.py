"""The ``domino`` key source: the run token a Domino run serves at a local URL.

On Domino each App sits behind a proxy that expects the run's token. The token
is asked for at the ``domino`` block's ``token_url``, held for its
``token_ttl_s``, and asked for again after that or once a call carrying it was
refused (the proxy's 401 for an expired token). Asking is a GET under the
shared ``http`` block's timeout, retries and backoff, which retry what the
retrying post of an endpoint retries (a transport error, a 5xx, a 429) and fail
at once on any other answer; it blocks, so code that awaits calls it off the
event loop. The block ``{source: domino}`` reads no other key.

Example:
    >>> source = DominoKeySource(httpx.Client(), token_url=url, ttl_s=240.0,
    ...                          policy=policy)
    >>> source.key(may_ask=True)
    'eyJ...'
"""

from __future__ import annotations

import threading
import time
from http import HTTPStatus
from typing import NamedTuple

import httpx

from ....domain.errors import RagDependencyError, describe_error
from ....infrastructure.http.http_policy import HttpPolicy
from ..base_key_source import KEY_SOURCES, BaseKeySource

# The error code of a run token that could not be had, as before key sources.
_TOKEN_COMPONENT = "domino_access_token"
# Retried like a 5xx, as the retrying post retries it: a rate limit lifts.
_RATE_LIMITED = HTTPStatus.TOO_MANY_REQUESTS


class DominoKeySource(BaseKeySource):
    """The run token, asked for at its URL and held for a while.

    One is shared by every endpoint of a process, so two calls refused with
    the same token cost one new ask.

    Args:
        client: The blocking HTTP client the token is asked for with.
        token_url: Where the run serves its token.
        ttl_s: Seconds a token is held before it is asked for again.
        policy: Timeout, retries and backoff of asking (the shared ``http``
            block's).

    Example:
        >>> DominoKeySource(client, token_url=url, ttl_s=240.0,
        ...                 policy=policy).key(may_ask=False) is None
        True
    """

    def __init__(
        self, client: httpx.Client, *, token_url: str, ttl_s: float, policy: HttpPolicy
    ) -> None:
        self._client, self._url, self._ttl_s = client, token_url, ttl_s
        self._policy = policy
        self._held: str | None = None
        self._asked_at = 0.0
        # Threads ask at once when the token expires: one asks, the others wait
        # and take its token. The state's own lock is never held while asking,
        # so a refusal, told on the event loop, does not wait on the network.
        self._asking = threading.Lock()
        self._state = threading.Lock()

    def key(self, *, may_ask: bool) -> str | None:
        """Give the run token held, asking for a new one when it is stale.

        Args:
            may_ask: Whether it may ask the token URL (a blocking call).

        Returns:
            The token; None when it holds none still fresh and may not ask.

        Raises:
            RagDependencyError: Under ``domino_access_token``, naming the token
                URL, when every ask failed.

        Example:
            >>> source.key(may_ask=True)
            'eyJ...'
        """
        fresh = self._fresh()
        if fresh is not None or not may_ask:
            return fresh
        with self._asking:
            return self._fresh() or self._ask()

    def refused(self, key: str) -> None:
        """Forget the token if it is still ``key``, so the next call asks again.

        Args:
            key: The token the refused call carried.

        Example:
            >>> source.refused("eyJ...")
        """
        with self._state:
            if self._held == key:
                self._held = None

    def _fresh(self) -> str | None:
        with self._state:
            if time.monotonic() - self._asked_at < self._ttl_s:
                return self._held
            return None

    def described(self) -> str:
        """Name the token URL, and the new ask a refusal brings.

        Returns:
            The clause an error of a refused call quotes.

        Example:
            >>> source.described()
            'its key is the Domino run token served at http://...'
        """
        return (
            f"its key is the Domino run token served at {self._url}, asked for "
            "again after each refusal"
        )

    def _ask(self) -> str:
        # What the retrying post retries is tried again; any other answer of
        # the token URL fails at once, as a retry would not change it.
        asked = _Asked("", "no attempt made", may_retry=True)
        for attempt in range(self._policy.retries + 1):
            time.sleep(self._policy.backoff_before(attempt))
            asked = self._attempt()
            if asked.token:
                return self._hold(asked.token)
            if not asked.may_retry:
                break
        raise RagDependencyError(_TOKEN_COMPONENT, asked.detail, target=self._url)

    def _hold(self, token: str) -> str:
        with self._state:
            self._held, self._asked_at = token, time.monotonic()
        return token

    def _attempt(self) -> _Asked:
        # The token, or why there is none (never a token's value).
        try:
            response = self._client.get(self._url, timeout=self._policy.timeout_s)
        except httpx.TransportError as exc:
            return _Asked("", describe_error(exc), may_retry=True)
        token, status = response.text.strip(), response.status_code
        if status == HTTPStatus.OK and token:
            return _Asked(token, "", may_retry=False)
        detail = (
            f"HTTP {status}: {response.text[:200]!r} "
            "(expected 200 with the token as the body)"
        )
        passing = status >= HTTPStatus.INTERNAL_SERVER_ERROR or status == _RATE_LIMITED
        return _Asked("", detail, may_retry=passing)


class _Asked(NamedTuple):
    # One ask of the token URL: the token, or why none came and whether a
    # retry may bring one.
    token: str
    detail: str
    may_retry: bool


# Registered by call: the decorator's return type would hide its constructor
# from the type checkers, and the composition root builds it by its class.
KEY_SOURCES.register("domino")(DominoKeySource)
