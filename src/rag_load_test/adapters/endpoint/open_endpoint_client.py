"""Build the client of one endpoint, with a bearer token of its own.

Each endpoint gets its own ``BearerToken`` (its own key, so no shared cache)
and ``ApiEndpointClient``. The HTTP client is shared and not owned here.

Example:
    >>> tokens = RunTokenSource(url="http://localhost:8899/access-token",
    ...                         ttl_s=240.0)
    >>> open_endpoint_client(client, endpoint, component="llm", tokens=tokens,
    ...                      policy=policy)
"""

from __future__ import annotations

from typing import NamedTuple

import httpx

from ...domain.api_endpoint import ApiEndpoint
from ..http.http_policy import HttpPolicy
from .api_endpoint_client import ApiEndpointClient
from .bearer_token import BearerToken


class RunTokenSource(NamedTuple):
    """Where the Domino run token is fetched, and how long one is reused.

    Attributes:
        url: The run token's URL (Domino key only).
        ttl_s: Seconds a fetched run token is reused.
    """

    url: str
    ttl_s: float


def open_endpoint_client(
    client: httpx.AsyncClient,
    endpoint: ApiEndpoint,
    *,
    component: str,
    tokens: RunTokenSource,
    policy: HttpPolicy,
) -> ApiEndpointClient:
    """Build the client of one endpoint, with a bearer token of its own.

    Args:
        client: The shared HTTP client.
        endpoint: The route, its API format, its key and its model.
        component: ``embedder``, ``reranker`` or ``llm``: names the errors.
        tokens: Where the Domino run token comes from, for the ``domino`` key.
        policy: Timeout, retries and backoff of every call.

    Returns:
        The endpoint client.

    Example:
        >>> open_endpoint_client(client, endpoint, component="llm",
        ...                      tokens=tokens, policy=policy).described()
        'gpt-4o-mini at https://api.openai.com/v1/chat/completions'
    """
    token = BearerToken(
        client,
        endpoint.api_key,
        token_url=tokens.url,
        ttl_s=tokens.ttl_s,
        policy=policy,
    )
    return ApiEndpointClient(
        client, endpoint, component=component, token=token, policy=policy
    )
