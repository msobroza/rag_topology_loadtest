"""BaseEmbedder over an HTTP endpoint that speaks an embeddings API format.

The query prefix goes on here, before the request: an embeddings format has no
field for it, and a model server of ours embeds what it receives as it is, so
the text on the wire is the text the model encodes.

Example:
    >>> embedder = ApiEmbedder(endpoint_client, EMBEDDING_API_FORMATS.get("openai")(),
    ...                        batch_size=64, query_prefix="query: ")
    >>> vectors = await embedder.embed_documents(["vacation policy", "expense report"])
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

import httpx
from pydantic import ValidationError

from ....domain.endpoints.api_endpoint import ApiEndpoint
from ....domain.errors import RagDependencyError, describe_error
from ....infrastructure.endpoint.api_endpoint_client import ApiEndpointClient
from ....infrastructure.endpoint.open_endpoint_client import (
    RunTokenSource,
    open_endpoint_client,
)
from ....infrastructure.http.http_policy import HttpPolicy
from ..base_embedder import BaseEmbedder
from .base_embedding_api_format import EMBEDDING_API_FORMATS, BaseEmbeddingApiFormat

_Send = Callable[[dict[str, object]], Awaitable[object]]
# What the readiness request embeds: one short text, the cheapest real call.
_READY_TEXTS = ("ready",)


class ApiEmbedder(BaseEmbedder):
    """Embeds texts on one endpoint, ``batch_size`` texts per request.

    Args:
        endpoint_client: The endpoint every request goes to (component
            ``embedder``).
        api_format: Builds each request body and reads each answer.
        batch_size: Texts per request.
        query_prefix: Put in front of a query, never of a text to store; "" for a
            model that takes none.

    Example:
        >>> embedder = ApiEmbedder(endpoint_client, OpenAIEmbeddingsFormat(),
        ...                        batch_size=64, query_prefix="query: ")
        >>> await embedder.embed_documents(["a", "b"])
        [[0.1, ...], [0.3, ...]]
    """

    def __init__(
        self,
        endpoint_client: ApiEndpointClient,
        api_format: BaseEmbeddingApiFormat,
        *,
        batch_size: int,
        query_prefix: str,
    ) -> None:
        self._endpoint_client, self._api_format = endpoint_client, api_format
        self._batch_size, self._query_prefix = batch_size, query_prefix
        self.model_name = endpoint_client.endpoint.model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts as they are, ``batch_size`` per request.

        Args:
            texts: The texts to embed; none sends no request.

        Returns:
            One vector per text, in the same order.

        Raises:
            RagDependencyError: If a request fails, or an answer is not the
                format's shape or has not one vector per text sent.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embed_batch(batch, self._endpoint_client.post))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query, the query prefix put in front before it is sent.

        Args:
            text: The query, as the user asked it.

        Returns:
            Its vector.

        Raises:
            RagDependencyError: If the request fails, or the answer is not the
                format's shape or has not one vector.

        Example:
            >>> vector = await embedder.embed_query("How many vacation days?")
        """
        query = self._query_prefix + text
        [vector] = await self._embed_batch([query], self._endpoint_client.post)
        return vector

    async def ready(self) -> tuple[bool, str]:
        """Embed one short text, once and without retry, to see the endpoint work.

        Readiness is one real minimal request: on a metered API it is billed like
        any other.

        Returns:
            ``True`` and ``"<model> at <url>"``; or ``False`` and the error, when
            the request fails or the answer is malformed. It never raises.
        """
        try:
            await self._embed_batch(_READY_TEXTS, self._endpoint_client.post_once)
        except RagDependencyError as exc:
            return False, describe_error(exc)
        return True, self._endpoint_client.described()

    async def _embed_batch(
        self, batch: Sequence[str], send: _Send
    ) -> list[list[float]]:
        # One request; a body of the wrong shape or count is a malformed answer.
        body = self._api_format.request_body(self.model_name, batch)
        answer = await send(body)
        try:
            vectors = self._api_format.vectors(answer)
        except ValidationError as exc:
            raise self._endpoint_client.malformed(exc) from exc
        if len(vectors) != len(batch):
            problem = ValueError(
                f"{len(vectors)} vectors for {len(batch)} texts sent "
                "(expected one per text)"
            )
            raise self._endpoint_client.malformed(problem)
        return vectors


def open_api_embedder(
    client: httpx.AsyncClient,
    endpoint: ApiEndpoint,
    *,
    tokens: RunTokenSource,
    policy: HttpPolicy,
    batch_size: int,
    query_prefix: str,
) -> ApiEmbedder:
    """Build the embedder of one embeddings endpoint.

    Args:
        client: The shared HTTP client.
        endpoint: The embeddings route, its API format, its key and its model.
        tokens: Where the Domino run token comes from, for the ``domino`` key.
        policy: Timeout, retries and backoff of every call.
        batch_size: Texts per request.
        query_prefix: Put in front of a query, never of a text to store.

    Returns:
        The embedder.

    Raises:
        ValueError: If ``endpoint.api_format`` names no embeddings API format.

    Example:
        >>> open_api_embedder(client, endpoint, tokens=tokens, policy=policy,
        ...                   batch_size=64, query_prefix="query: ")
    """
    api_format = EMBEDDING_API_FORMATS.get(endpoint.api_format)()
    endpoint_client = open_endpoint_client(
        client, endpoint, component="embedder", tokens=tokens, policy=policy
    )
    return ApiEmbedder(
        endpoint_client, api_format, batch_size=batch_size, query_prefix=query_prefix
    )
