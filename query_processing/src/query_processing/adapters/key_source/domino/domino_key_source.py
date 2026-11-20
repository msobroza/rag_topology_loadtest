"""The ``domino`` key source: the run token a Domino run serves at a local URL.

On Domino each App sits behind a proxy that expects the run's token. The token
is asked for at the ``domino`` block's ``token_url``, held for its
``token_ttl_s``, and asked for again after that or once a call carrying it was
refused (the proxy's 401 for an expired token). Asking is a GET under the
shared ``http`` block's timeout, retries and backoff; it blocks, so code that
awaits calls it off the event loop.

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

import httpx

from ....domain.errors import RagDependencyError, describe_error
from ....infrastructure.http.http_policy import HttpPolicy
from ..base_key_source import KEY_SOURCES, BaseKeySource

# The error code of a run token that could not be had, as before key sources.
_TOKEN_COMPONENT = "domino_access_token"


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

    def _ask(self) -> str:
        # Every failed ask may be a passing one: the token URL is the run's own.
        detail = "no attempt made"
        for attempt in range(self._policy.retries + 1):
            time.sleep(self._policy.backoff_before(attempt))
            token, detail = self._attempt()
            if token:
                return self._hold(token)
        raise RagDependencyError(_TOKEN_COMPONENT, detail, target=self._url)

    def _hold(self, token: str) -> str:
        with self._state:
            self._held, self._asked_at = token, time.monotonic()
        return token

    def _attempt(self) -> tuple[str, str]:
        # The token and "", or "" and why there is none (never a token's value).
        try:
            response = self._client.get(self._url, timeout=self._policy.timeout_s)
        except httpx.TransportError as exc:
            return "", describe_error(exc)
        token = response.text.strip()
        if response.status_code == HTTPStatus.OK and token:
            return token, ""
        return "", (
            f"HTTP {response.status_code}: {response.text[:200]!r} "
            "(expected 200 with the token as the body)"
        )


# Registered by call: the decorator's return type would hide its constructor
# from the type checkers, and the composition root builds it by its class.
KEY_SOURCES.register("domino")(DominoKeySource)
