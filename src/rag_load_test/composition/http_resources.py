"""What every call out shares: one HTTP client, its policy and the run tokens.

The client is created on first need and closed with the scope. The policy and
the run token source are plain values, read from the settings once. Where a
component on the ``served`` backend is called is resolved here too, since its
key depends on where the process runs: on Domino a model server of ours is
another App, behind a proxy that wants the run's token.

Example:
    >>> http = HttpResources(settings, scope)
    >>> client = await http.client()
"""

from __future__ import annotations

import httpx

from ..config.component_settings import ModelComponent
from ..config.rag_settings import RagSettings
from ..domain.api_endpoint import DOMINO_RUN_TOKEN, ApiEndpoint
from ..domain.standard_formats import STANDARD_API_FORMATS
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
        on_domino: Whether the process runs as a Domino App, where a served
            component is called with the run token (``rag-serve --domino``).

    Example:
        >>> HttpResources(settings, scope).policy.timeout_s
        30.0
    """

    def __init__(
        self,
        settings: RagSettings,
        scope: ResourceScope,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        on_domino: bool = False,
    ) -> None:
        self.policy = HttpPolicy(
            settings.http_timeout_s, settings.http_retries, settings.http_backoff_s
        )
        self.run_tokens = RunTokenSource(
            settings.domino_token_url, settings.domino_token_ttl_s
        )
        # Off Domino a model server of ours takes no key.
        self._served_key = DOMINO_RUN_TOKEN if on_domino else ""
        self._scope = scope
        self._transport = transport

    def served_endpoint(self, component: ModelComponent) -> ApiEndpoint:
        """Return where a component on the ``served`` backend is called.

        Args:
            component: The component, its ``served`` block giving the route.

        Returns:
            The route, the standard format of the component's type, the run
            token on Domino and no key elsewhere, and the alias as the model
            name: a model server of ours serves a model under its alias.

        Example:
            >>> http.served_endpoint(settings.components["reranker"])
            ApiEndpoint(url='http://reranker-app/v3/rerank', api_format='cohere',
                        api_key='domino', model='crossencoder-camembert-base-mmarcoFR')
        """
        api_format = STANDARD_API_FORMATS[component.type]
        url = component.served.url
        return ApiEndpoint(url, api_format, self._served_key, component.alias)

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
