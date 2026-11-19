"""What every call out shares: one HTTP client, the default policy and the run tokens.

The client is created on first need and closed with the scope. The default
policy and the run token source are plain values, read from the settings once.
How patiently a component on ``served`` or ``api`` is called is resolved here:
its own ``http`` block, each value it leaves out taken from the default. Where
a component on the ``served`` backend is called is resolved here too, since its
key depends on where the process runs: on Domino a model server of ours is
another App, behind a proxy that wants the run's token. The client's pool
and its HTTP version are the top-level ``http`` block's alone.

Example:
    >>> http = HttpResources(settings, scope)
    >>> client = await http.client()
"""

from __future__ import annotations

import dataclasses
import importlib.util
from collections.abc import Callable

import httpx

from ..config.components.component_settings import ModelComponent
from ..config.components.registered_component_settings import (
    RegisteredComponentSettings,
)
from ..config.workflow_settings import WorkflowSettings
from ..domain.endpoints.api_endpoint import DOMINO_RUN_TOKEN, ApiEndpoint
from ..domain.endpoints.standard_formats import STANDARD_API_FORMATS
from ..infrastructure.endpoint.open_endpoint_client import RunTokenSource
from ..infrastructure.http.http_policy import HttpPolicy
from .resource_scope import OWNED, ResourceScope

ClientFactory = Callable[..., httpx.AsyncClient]


def h2_is_installed() -> bool:
    """Return whether the ``h2`` package, which httpx's HTTP/2 needs, is importable.

    Returns:
        True when ``h2`` (the ``httpx[http2]`` extra) can be imported.

    Example:
        >>> h2_is_installed()
        False
    """
    return importlib.util.find_spec("h2") is not None


class HttpResources:
    """The shared HTTP client, the default HTTP policy and where run tokens come from.

    Args:
        settings: The default timeout, retries and backoff and the client's
            pool (its ``http`` block), and the Domino token's URL and lifetime
            (its ``domino`` block).
        scope: Opens the client once and closes it.
        transport: The client's transport; None for the real network (tests
            pass an ``httpx.MockTransport``).
        on_domino: Whether the process runs as a Domino App, where a served
            component is called with the run token (``qp-serve --domino``).
        client_factory: Builds the shared client from its keywords; tests
            pass one that records them.
        h2_installed: Whether HTTP/2's ``h2`` package is importable; tests
            pass a fixed answer.

    Raises:
        ValueError: ``http2`` is on and ``h2`` is not installed.

    Example:
        >>> HttpResources(settings, scope).policy.timeout_s
        30.0
    """

    def __init__(
        self,
        settings: WorkflowSettings,
        scope: ResourceScope,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        on_domino: bool = False,
        client_factory: ClientFactory = httpx.AsyncClient,
        h2_installed: Callable[[], bool] = h2_is_installed,
    ) -> None:
        # Every component's default, and the policy of the calls that are no
        # component's, such as the Domino hardware-tier lookup.
        http, domino = settings.http, settings.domino
        # The client is built on first need, where httpx would refuse it; refuse
        # it at start-up instead.
        if http.http2 and not h2_installed():
            raise ValueError(
                "QP_HTTP__HTTP2=true needs the h2 package: install the "
                "httpx[http2] extra, or set QP_HTTP__HTTP2=false"
            )
        self._http = http
        self._client_factory = client_factory
        self.policy = HttpPolicy(http.timeout_s, http.retries, http.backoff_s)
        self.run_tokens = RunTokenSource(domino.token_url, domino.token_ttl_s)
        # Off Domino a model server of ours takes no key.
        self._served_key = DOMINO_RUN_TOKEN if on_domino else ""
        self._scope = scope
        self._transport = transport

    def policy_of(
        self, component: ModelComponent | RegisteredComponentSettings
    ) -> HttpPolicy:
        """Return how patiently a component on ``served`` or ``api`` is called.

        Args:
            component: The component, its ``http`` block giving its own values.

        Returns:
            The timeout, retries and backoff its ``http`` block sets, and the
            default policy's for each it leaves out.

        Example:
            >>> http.policy_of(settings.components["llm"])
            HttpPolicy(timeout_s=120.0, retries=2, backoff_s=0.2)
        """
        own = component.http.model_dump(exclude_none=True)
        return dataclasses.replace(self.policy, **own)

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

    def call_endpoint(
        self, name: str, component: RegisteredComponentSettings
    ) -> ApiEndpoint:
        """Return where a registered component on the ``served`` backend is called.

        Args:
            name: The component's name, the model name its model server serves
                it under.
            component: The component, its ``served`` block giving the call route.

        Returns:
            The call route, its component type as the API format's name, the
            run token on Domino and no key elsewhere, and its name as the model.

        Example:
            >>> http.call_endpoint("shouter", settings.components["shouter"])
            ApiEndpoint(url='http://shouter-app/call/shouter', api_format='shouter',
                        api_key='', model='shouter')
        """
        url, api_format = component.served.url, component.type
        return ApiEndpoint(url, api_format, self._served_key, name)

    async def client(self) -> httpx.AsyncClient:
        """Return the shared HTTP client, created on the first request.

        Returns:
            The client, closed when the scope ends.

        Example:
            >>> client = await http.client()
        """
        return await self._scope.once("http_client", OWNED, self._start_client)

    async def _start_client(self) -> httpx.AsyncClient:
        http = self._http
        limits = httpx.Limits(
            max_connections=http.max_connections,
            max_keepalive_connections=http.max_keepalive_connections,
            keepalive_expiry=http.keepalive_expiry_s,
        )
        client = self._client_factory(
            timeout=self.policy.timeout_s,
            limits=limits,
            http2=http.http2,
            transport=self._transport,
        )
        self._scope.exit_stack("http_client").push_async_callback(client.aclose)
        return client
