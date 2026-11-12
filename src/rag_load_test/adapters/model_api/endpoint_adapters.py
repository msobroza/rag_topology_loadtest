"""Build the ``api`` embedder, reranker and LLM of one endpoint from plain values.

Each endpoint gets its own ``BearerToken`` (its own key, so no shared cache) and
``ApiEndpointClient``; the API format comes from the registry by the
endpoint's ``api_format`` name. The HTTP client is shared and not owned here.

Example:
    >>> tokens = RunTokenSource(url="http://localhost:8899/access-token",
    ...                         ttl_s=240.0)
    >>> reranker = open_api_reranker(client, endpoint, tokens=tokens,
    ...                              policy=policy)
"""

from __future__ import annotations

from typing import NamedTuple

import httpx

from ...domain.api_endpoint import ApiEndpoint
from ..http.http_policy import HttpPolicy
from .api_embedder import ApiEmbedder
from .api_endpoint_client import ApiEndpointClient
from .api_formats import embedding_api_format, generation_api_format, rerank_api_format
from .api_llm import ApiLLM
from .api_reranker import ApiReranker
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


def open_api_embedder(
    client: httpx.AsyncClient,
    endpoint: ApiEndpoint,
    *,
    tokens: RunTokenSource,
    policy: HttpPolicy,
    batch_size: int,
) -> ApiEmbedder:
    """Build the embedder of one embeddings endpoint.

    Args:
        client: The shared HTTP client.
        endpoint: The embeddings route, its API format, its key and its model.
        tokens: Where the Domino run token comes from, for the ``domino`` key.
        policy: Timeout, retries and backoff of every call.
        batch_size: Texts per request.

    Returns:
        The embedder.

    Raises:
        ValueError: If ``endpoint.api_format`` names no embeddings API format.

    Example:
        >>> open_api_embedder(client, endpoint, tokens=tokens, policy=policy,
        ...                   batch_size=64)
    """
    api_format = embedding_api_format(endpoint.api_format)
    endpoint_client = open_endpoint_client(
        client, endpoint, component="embedder", tokens=tokens, policy=policy
    )
    return ApiEmbedder(endpoint_client, api_format, batch_size=batch_size)


def open_api_reranker(
    client: httpx.AsyncClient,
    endpoint: ApiEndpoint,
    *,
    tokens: RunTokenSource,
    policy: HttpPolicy,
) -> ApiReranker:
    """Build the reranker of one rerank endpoint.

    Args:
        client: The shared HTTP client.
        endpoint: The rerank route, its API format, its key and its model.
        tokens: Where the Domino run token comes from, for the ``domino`` key.
        policy: Timeout, retries and backoff of every call.

    Returns:
        The reranker.

    Raises:
        ValueError: If ``endpoint.api_format`` names no rerank API format.

    Example:
        >>> open_api_reranker(client, endpoint, tokens=tokens, policy=policy)
    """
    api_format = rerank_api_format(endpoint.api_format)
    endpoint_client = open_endpoint_client(
        client, endpoint, component="reranker", tokens=tokens, policy=policy
    )
    return ApiReranker(endpoint_client, api_format)


def open_api_llm(
    client: httpx.AsyncClient,
    endpoint: ApiEndpoint,
    *,
    tokens: RunTokenSource,
    policy: HttpPolicy,
    temperature: float,
) -> ApiLLM:
    """Build the LLM of one generation endpoint.

    Args:
        client: The shared HTTP client.
        endpoint: The generation route, its API format, its key and its model.
        tokens: Where the Domino run token comes from, for the ``domino`` key.
        policy: Timeout, retries and backoff of every call; generation has a
            longer timeout than the model servers.
        temperature: The sampling temperature sent with every request.

    Returns:
        The LLM.

    Raises:
        ValueError: If ``endpoint.api_format`` names no generation API format.

    Example:
        >>> open_api_llm(client, endpoint, tokens=tokens, policy=policy,
        ...              temperature=0.0)
    """
    api_format = generation_api_format(endpoint.api_format)
    endpoint_client = open_endpoint_client(
        client, endpoint, component="llm", tokens=tokens, policy=policy
    )
    return ApiLLM(endpoint_client, api_format, temperature=temperature)
