"""The HTTP plumbing shared by every model endpoint: auth, retries, error shape.

Calls follow one :class:`HttpPolicy`: each attempt times out after
``timeout_s``; transport errors, 5xx and 429 (rate limited) are retried, and so
is 401 when the key is the Domino run token, after the token is dropped. Any
other 4xx, a rejected static key included, fails at once.

Example:
    >>> endpoint_client = ApiEndpointClient(client, endpoint, component="reranker",
    ...                                     token=token, policy=policy)
    >>> body = await endpoint_client.post({"model": "m", "query": "q",
    ...                                    "documents": ["a"]})
"""

from __future__ import annotations

import json
from http import HTTPStatus

import httpx

from ...domain.api_endpoint import ApiEndpoint
from ...domain.errors import RagDependencyError
from ..shared.http_policy import HttpPolicy
from ..shared.json_post import RetryingJsonPost, describe_malformed
from .bearer_token import BearerToken

# Retried like a 5xx whatever the key: a rate limit lifts by itself.
_ALWAYS_RETRIED = frozenset({HTTPStatus.TOO_MANY_REQUESTS})


class ApiEndpointClient:
    """POSTs JSON to one model endpoint and shapes every failure the same way.

    Args:
        client: The shared HTTP client every call goes through.
        endpoint: The route, its model and its key.
        component: Names the dependency in every error: ``embedder``,
            ``reranker`` or ``llm`` (they become the workflow's error codes).
        token: Supplies the ``Authorization`` header of every attempt.
        policy: Timeout, retries and backoff of ``post``; ``post_once`` keeps the
            timeout and makes one attempt.

    Example:
        >>> endpoint_client = ApiEndpointClient(client, endpoint, component="llm",
        ...                                     token=token, policy=policy)
        >>> endpoint_client.described()
        'gpt-4o-mini at https://gateway/v1/chat/completions'
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        endpoint: ApiEndpoint,
        *,
        component: str,
        token: BearerToken,
        policy: HttpPolicy,
    ) -> None:
        self._endpoint, self._component, self._token = endpoint, component, token
        one_attempt = HttpPolicy(timeout_s=policy.timeout_s, retries=0, backoff_s=0.0)
        self._retrying = self._poster(client, policy)
        self._once = self._poster(client, one_attempt)

    def _poster(
        self, client: httpx.AsyncClient, policy: HttpPolicy
    ) -> RetryingJsonPost:
        # Both posters retry (or, with zero retries, report) the same statuses.
        return RetryingJsonPost(
            client,
            policy,
            headers=self._token.headers,
            retry_statuses=_ALWAYS_RETRIED | self._token.retry_statuses,
            on_retry_status=self._drop_stale_token,
        )

    @property
    def endpoint(self) -> ApiEndpoint:
        """The endpoint every call goes to.

        Returns:
            The route, its API format, its key and its model.
        """
        return self._endpoint

    async def post(self, payload: dict[str, object]) -> object:
        """POST ``payload`` under the policy, retrying what a retry may fix.

        Args:
            payload: The JSON body.

        Returns:
            The decoded JSON body of the first answer below 400.

        Raises:
            RagDependencyError: If every attempt failed, the endpoint refused the
                request, or the answer is not JSON.
        """
        return await self._send(self._retrying, payload)

    async def post_once(self, payload: dict[str, object]) -> object:
        """POST ``payload`` once, with the policy's timeout and no retry.

        Args:
            payload: The JSON body.

        Returns:
            The decoded JSON body, if the answer is below 400.

        Raises:
            RagDependencyError: If the attempt failed or the answer is not JSON.
        """
        return await self._send(self._once, payload)

    def malformed(self, exc: Exception) -> RagDependencyError:
        """Build the error for an answer that is not the expected shape.

        Args:
            exc: What broke reading the answer, such as a ``ValidationError``.

        Returns:
            The error to raise, targeting the endpoint's URL.

        Example:
            >>> raise endpoint_client.malformed(exc) from exc
        """
        detail = f"malformed response: {describe_malformed(exc)}"
        return RagDependencyError(self._component, detail, target=self._endpoint.url)

    def described(self) -> str:
        """Name the endpoint for a status line.

        Returns:
            ``"<model> at <url>"``.

        Example:
            >>> endpoint_client.described()
            'rerank-v3.5 at https://api.cohere.com/v2/rerank'
        """
        return f"{self._endpoint.model} at {self._endpoint.url}"

    async def _send(self, poster: RetryingJsonPost, payload: object) -> object:
        # A 2xx whose body is not JSON is a malformed answer, not a retry.
        try:
            return await poster.post(self._component, self._endpoint.url, payload)
        except json.JSONDecodeError as exc:
            raise self.malformed(exc) from exc

    def _drop_stale_token(self, status: int) -> None:
        # Only reached for 401 with the Domino key: its run token expired.
        if status == HTTPStatus.UNAUTHORIZED:
            self._token.invalidate()
