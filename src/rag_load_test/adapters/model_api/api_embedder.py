"""BaseEmbedder over an HTTP endpoint that speaks an embeddings API format.

Example:
    >>> embedder = ApiEmbedder(endpoint_client, embedding_api_format("openai"),
    ...                        batch_size=64)
    >>> vectors = await embedder.embed(["vacation policy", "expense report"])
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from pydantic import ValidationError

from ...domain.errors import RagDependencyError, describe_error
from ...domain.ports import BaseEmbedder
from .api_endpoint_client import ApiEndpointClient
from .api_format_bases import BaseEmbeddingApiFormat

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

    Example:
        >>> embedder = ApiEmbedder(endpoint_client, OpenAIEmbeddingsFormat(),
        ...                        batch_size=64)
        >>> await embedder.embed(["a", "b"])
        [[0.1, ...], [0.3, ...]]
    """

    def __init__(
        self,
        endpoint_client: ApiEndpointClient,
        api_format: BaseEmbeddingApiFormat,
        *,
        batch_size: int,
    ) -> None:
        self._endpoint_client, self._api_format = endpoint_client, api_format
        self._batch_size = batch_size
        self.model_name = endpoint_client.endpoint.model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts, ``batch_size`` per request.

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
