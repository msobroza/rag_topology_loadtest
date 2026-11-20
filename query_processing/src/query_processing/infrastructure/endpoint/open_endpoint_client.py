"""Build the client of one endpoint, with the key source it is given or none.

The key source is built by the composition root and may be shared by several
endpoints (the process's one ``domino`` source); the HTTP client is shared and
not owned here.

Example:
    >>> open_endpoint_client(client, endpoint, component="llm",
    ...                      key_source=None, policy=policy)
"""

from __future__ import annotations

import httpx

from ...domain.endpoints.api_endpoint import ApiEndpoint
from ..http.http_policy import HttpPolicy
from .api_endpoint_client import ApiEndpointClient
from .bearer_key import BearerKey, KeySupplier


def open_endpoint_client(
    client: httpx.AsyncClient,
    endpoint: ApiEndpoint,
    *,
    component: str,
    key_source: KeySupplier | None,
    policy: HttpPolicy,
) -> ApiEndpointClient:
    """Build the client of one endpoint, its key from ``key_source``.

    Args:
        client: The shared HTTP client.
        endpoint: The route, its API format and its model.
        component: ``embedder``, ``reranker`` or ``llm``: names the errors.
        key_source: Where the endpoint's API key comes from; None sends no key.
        policy: Timeout, retries and backoff of every call.

    Returns:
        The endpoint client.

    Example:
        >>> open_endpoint_client(client, endpoint, component="llm",
        ...                      key_source=None, policy=policy).described()
        'gpt-4o-mini at https://api.openai.com/v1/chat/completions'
    """
    key = BearerKey(key_source) if key_source is not None else None
    return ApiEndpointClient(
        client, endpoint, component=component, key=key, policy=policy
    )
