"""What every call out shares: one HTTP client, its policy and the run tokens.

The client is created on first need and closed with the scope. The policy and
the run token source are plain values, read from the settings once.

Example:
    >>> http = HttpResources(settings, scope)
    >>> client = await http.client()
"""

from __future__ import annotations

import httpx

from ..config.rag_settings import RagSettings
from ..infrastructure.endpoint.open_endpoint_client import RunTokenSource
from ..infrastructure.http.http_policy import HttpPolicy
from .resource_scope import OWNED, ResourceScope


class HttpResources:
    """The shared HTTP client, the HTTP policy and where run tokens come from.

    Args:
        settings: The timeout, retries, backoff and the Domino token URL.
        scope: Opens the client once and closes it.
        transport: The client's transport; None for the real network (tests
            pass an ``httpx.MockTransport``).

    Example:
        >>> HttpResources(settings, scope).policy.timeout_s
        30.0
    """

    def __init__(
        self,
        settings: RagSettings,
        scope: ResourceScope,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.policy = HttpPolicy(
            settings.http_timeout_s, settings.http_retries, settings.http_backoff_s
        )
        self.run_tokens = RunTokenSource(
            settings.domino_token_url, settings.domino_token_ttl_s
        )
        self._scope = scope
        self._transport = transport

    async def client(self) -> httpx.AsyncClient:
        """Return the shared HTTP client, created on the first request.

        Returns:
            The client, closed when the scope ends.

        Example:
            >>> client = await http.client()
        """
        return await self._scope.once("http_client", OWNED, self._start_client)

    async def _start_client(self) -> httpx.AsyncClient:
        timeout = self.policy.timeout_s
        client = httpx.AsyncClient(timeout=timeout, transport=self._transport)
        self._scope.exit_stack("http_client").push_async_callback(client.aclose)
        return client
