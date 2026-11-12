"""The ``Authorization`` header of a model endpoint: none, a static key, or Domino's.

On Domino each App sits behind a proxy that expects the run's bearer token; it
is fetched from a local URL and reused for a while, and dropped when the proxy
answers 401 (an expired run token).

Example:
    >>> token = BearerToken(client, "domino", token_url=url, ttl_s=240.0,
    ...                     policy=policy)
    >>> await token.headers()
    {'Authorization': 'Bearer <run token>'}
"""

from __future__ import annotations

import time
from http import HTTPStatus

import httpx

from ...domain.api_endpoint import DOMINO_RUN_TOKEN
from ...domain.errors import RagDependencyError, describe_error
from ..http.http_policy import HttpPolicy

_TOKEN_COMPONENT = "domino_access_token"


class BearerToken:
    """The ``Authorization`` header for every call to one endpoint.

    Args:
        client: The shared HTTP client the run token is fetched with.
        api_key: "" sends no header; ``DOMINO_RUN_TOKEN`` sends the run token;
            any other value is sent as a static bearer token.
        token_url: Where Domino's run token is fetched (Domino key only).
        ttl_s: Seconds a fetched run token is reused before it is fetched again.
        policy: Only its ``timeout_s`` is used: a token fetch is tried once, and a
            failed fetch fails the model call, which retries.

    Example:
        >>> token = BearerToken(client, "abc", token_url=url, ttl_s=240,
        ...                     policy=policy)
        >>> await token.headers()
        {'Authorization': 'Bearer abc'}
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        *,
        token_url: str,
        ttl_s: float,
        policy: HttpPolicy,
    ) -> None:
        self._client, self._api_key = client, api_key
        self._token_url, self._ttl_s = token_url, ttl_s
        self._timeout_s = policy.timeout_s
        self._cached: str | None = None
        self._fetched_at = 0.0

    @property
    def retry_statuses(self) -> frozenset[int]:
        """The statuses a fresh token may fix, so a call should retry them.

        Returns:
            ``{401}`` for the Domino run token (the proxy's answer to an expired
            one); empty otherwise, since a wrong static key does not fix itself.
        """
        if self._api_key == DOMINO_RUN_TOKEN:
            return frozenset({HTTPStatus.UNAUTHORIZED})
        return frozenset()

    async def headers(self) -> dict[str, str]:
        """Return the headers to send, fetching the run token when it is stale.

        Returns:
            ``{"Authorization": "Bearer <token>"}``, or ``{}`` when the key is "".

        Raises:
            RagDependencyError: If the Domino run token cannot be fetched
                (component ``domino_access_token``).
        """
        if not self._api_key:
            return {}
        is_domino = self._api_key == DOMINO_RUN_TOKEN
        token = await self._domino_token() if is_domino else self._api_key
        return {"Authorization": f"Bearer {token}"}

    def invalidate(self) -> None:
        """Drop the cached run token, so the next ``headers`` fetches a new one."""
        self._cached = None

    async def _domino_token(self) -> str:
        if self._cached and time.monotonic() - self._fetched_at < self._ttl_s:
            return self._cached
        url = self._token_url
        try:
            response = await self._client.get(url, timeout=self._timeout_s)
        except httpx.TransportError as exc:
            raise RagDependencyError(
                _TOKEN_COMPONENT, describe_error(exc), target=url
            ) from exc
        token = response.text.strip()
        if response.status_code != HTTPStatus.OK or not token:
            detail = (
                f"HTTP {response.status_code}: {response.text[:200]!r} "
                "(expected 200 with the token as the body)"
            )
            raise RagDependencyError(_TOKEN_COMPONENT, detail, target=url)
        self._cached, self._fetched_at = token, time.monotonic()
        return token
