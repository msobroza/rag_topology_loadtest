"""A hosted rerank route in the Cohere shape (the ``cohere`` and ``openai`` backends).

Cohere's own ``/v2/rerank`` and the ``/rerank`` routes of OpenAI-compatible
gateways (LiteLLM, vLLM, Jina, Infinity) share one shape, the one OVMS speaks
too, so the request and response models come from :mod:`..ovms.v3_contract`.
Calls follow one :class:`HttpPolicy`: they time out after ``timeout_s`` and are
retried on transport errors, 5xx and 429 (rate limited); any other 4xx,
including a rejected key, fails at once.

Example:
    >>> reranker = RerankApiReranker(client, "https://api.cohere.com/v2/rerank",
    ...                              "rerank-v3.5", api_key=key, policy=policy)
    >>> best = await reranker.rerank("vacation policy", candidates, top_n=5)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from http import HTTPStatus

import httpx
from pydantic import ValidationError

from ...domain.errors import RagDependencyError
from ...domain.passages import ScoredPassage
from ...domain.ports import RerankerPort
from ..ovms.v3_contract import RerankResponse, passages_from_rerank, rerank_request_for
from ..shared.http_policy import HttpPolicy
from ..shared.json_post import RetryingJsonPost, describe_malformed, no_headers

COMPONENT = "reranker"
# Retried like a 5xx: a rate limit lifts by itself, a wrong key (401/403) does not.
RERANK_RETRY_STATUSES = frozenset({HTTPStatus.TOO_MANY_REQUESTS})


class RerankApiReranker(RerankerPort):
    """RerankerPort over ``POST <url>`` with ``{model, query, documents, top_n}``.

    Args:
        client: The shared HTTP client every call goes through.
        url: The rerank route, in full.
        model: The model name sent with every request.
        api_key: Sent as ``Authorization: Bearer``; "" sends no header.
        policy: Timeout, retries and backoff of every call.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        model: str,
        *,
        api_key: str,
        policy: HttpPolicy,
    ) -> None:
        self._url, self._model, self._api_key = url, model, api_key
        headers = self._bearer if api_key else no_headers
        self._poster = RetryingJsonPost(
            client, policy, headers=headers, retry_statuses=RERANK_RETRY_STATUSES
        )

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        """Re-score the candidates on the hosted route and keep the best.

        Args:
            query: The question the passages are scored against.
            passages: The retrieval candidates; none sends no request.
            top_n: How many passages to keep.

        Returns:
            At most ``top_n`` passages, best first, with the route's score.

        Raises:
            RagDependencyError: If the route keeps failing, rejects the request,
                or answers with a body that is not the rerank shape or names an
                index outside the candidates.
        """
        if not passages:
            return []
        request = rerank_request_for(self._model, query, passages, top_n)
        try:
            body = await self._poster.post(COMPONENT, self._url, request.to_wire())
            response = RerankResponse.model_validate(body)
            return passages_from_rerank(response, passages, top_n)
        # JSONDecodeError: a non-JSON body; IndexError: an index out of range.
        except (json.JSONDecodeError, ValidationError, IndexError) as exc:
            detail = f"malformed response: {describe_malformed(exc)}"
            raise RagDependencyError(COMPONENT, detail, target=self._url) from exc

    async def ready(self) -> tuple[bool, str]:
        """Report ready without calling the route (a probe would be billed).

        Returns:
            ``True`` and the model with its route.
        """
        return True, f"{self._model} at {self._url}"

    async def _bearer(self) -> dict[str, str]:
        # The header supplier when a key is set; the poster calls it per attempt.
        return {"Authorization": f"Bearer {self._api_key}"}
