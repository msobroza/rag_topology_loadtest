"""BaseReranker over an HTTP endpoint that speaks a rerank API format.

Example:
    >>> reranker = ApiReranker(endpoint_client, RERANK_API_FORMATS.get("cohere")())
    >>> best = await reranker.rerank("vacation policy", candidates, top_n=5)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

import httpx
from pydantic import ValidationError
from query_processing_core.retrieval.vector_store_record import ScoredVectorStoreRecord

from ....domain.endpoints.api_endpoint import ApiEndpoint
from ....domain.errors import RagDependencyError, describe_error
from ....domain.retrieval.rag_result import top_by_score
from ....infrastructure.endpoint.api_endpoint_client import ApiEndpointClient
from ....infrastructure.endpoint.open_endpoint_client import open_endpoint_client
from ....infrastructure.http.http_policy import HttpPolicy
from ...key_source.base_key_source import BaseKeySource
from ..base_reranker import BaseReranker
from .base_rerank_api_format import RERANK_API_FORMATS, BaseRerankApiFormat

_Send = Callable[[dict[str, object]], Awaitable[object]]
# What the readiness request scores: one query and one document.
_READY_QUERY = "ready"
_READY_RECORDS = (
    ScoredVectorStoreRecord(
        text="ready", metadata={"id": "ready"}, retrieval_score=0.0
    ),
)


class ApiReranker(BaseReranker):
    """Re-scores retrieval candidates on one endpoint and keeps the best.

    Args:
        endpoint_client: The endpoint every request goes to (component
            ``reranker``).
        api_format: Builds each request body and reads each answer.

    Example:
        >>> reranker = ApiReranker(endpoint_client, CohereRerankFormat())
        >>> [p.id for p in await reranker.rerank("q", candidates, top_n=2)]
        ['p2', 'p0']
    """

    def __init__(
        self, endpoint_client: ApiEndpointClient, api_format: BaseRerankApiFormat
    ) -> None:
        self._endpoint_client, self._api_format = endpoint_client, api_format

    async def rerank(
        self, query: str, records: Sequence[ScoredVectorStoreRecord], top_n: int
    ) -> list[ScoredVectorStoreRecord]:
        """Re-score the candidates on the endpoint and keep the best.

        Args:
            query: The question the records are scored against.
            records: The retrieval candidates; none sends no request.
            top_n: How many records to keep.

        Returns:
            At most ``top_n`` records, best first, with the endpoint's score.

        Raises:
            RagDependencyError: If the request fails, or the answer is not the
                format's shape or names an index outside the records sent.
        """
        if not records:
            return []
        return await self._rerank(query, records, top_n, self._endpoint_client.post)

    async def ready(self) -> tuple[bool, str]:
        """Score one document, once and without retry, to see the endpoint work.

        Readiness is one real minimal request: on a metered API it is billed like
        any other.

        Returns:
            ``True`` and ``"<model> at <url>"``; or ``False`` and the error, when
            the request fails or the answer is malformed. It never raises.
        """
        send = self._endpoint_client.post_once
        try:
            await self._rerank(_READY_QUERY, _READY_RECORDS, 1, send)
        except RagDependencyError as exc:
            return False, describe_error(exc)
        return True, self._endpoint_client.described()

    async def _rerank(
        self,
        query: str,
        records: Sequence[ScoredVectorStoreRecord],
        top_n: int,
        send: _Send,
    ) -> list[ScoredVectorStoreRecord]:
        # One request; the answer's indexes name the records sent, in order.
        model = self._endpoint_client.endpoint.model
        documents = [record.text for record in records]
        answer = await send(
            self._api_format.request_body(model, query, documents, top_n)
        )
        try:
            scores = self._api_format.scores(answer)
            chosen = [_record_at(records, index) for index in scores]
        except (ValidationError, IndexError) as exc:
            raise self._endpoint_client.malformed(exc) from exc
        return top_by_score(chosen, list(scores.values()), top_n)


def _record_at(
    records: Sequence[ScoredVectorStoreRecord], index: int
) -> ScoredVectorStoreRecord:
    # Negative indexes would silently wrap around in Python: reject them too.
    if not 0 <= index < len(records):
        raise IndexError(
            f"rerank result index {index} is outside the {len(records)} documents "
            f"sent (expected 0..{len(records) - 1})"
        )
    return records[index]


def open_api_reranker(
    client: httpx.AsyncClient,
    endpoint: ApiEndpoint,
    *,
    key_source: BaseKeySource | None,
    policy: HttpPolicy,
) -> ApiReranker:
    """Build the reranker of one rerank endpoint.

    Args:
        client: The shared HTTP client.
        endpoint: The rerank route, its API format and its model.
        key_source: Where the endpoint's API key comes from; None sends no key.
        policy: Timeout, retries and backoff of every call.

    Returns:
        The reranker.

    Raises:
        ValueError: If ``endpoint.api_format`` names no rerank API format.

    Example:
        >>> open_api_reranker(client, endpoint, key_source=key_source, policy=policy)
    """
    api_format = RERANK_API_FORMATS.get(endpoint.api_format)()
    endpoint_client = open_endpoint_client(
        client, endpoint, component="reranker", key_source=key_source, policy=policy
    )
    return ApiReranker(endpoint_client, api_format)
