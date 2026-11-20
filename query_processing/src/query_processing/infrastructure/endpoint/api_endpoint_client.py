"""The HTTP plumbing shared by every model endpoint: auth, retries, error shape.

Calls follow one :class:`HttpPolicy`: each attempt times out after
``timeout_s``; transport errors, 5xx and 429 (rate limited) are retried, and so
is 401 when the endpoint is given a key source, after the source is told which
key was refused. A 401 with no key, a 403 and any other 4xx fail at once. When
the retries run out on a 401, the error says where the key comes from and how
it is changed, as its source says it; no error quotes a key sent.

Example:
    >>> endpoint_client = ApiEndpointClient(client, endpoint, component="reranker",
    ...                                     key=BearerKey(source), policy=policy)
    >>> body = await endpoint_client.post({"model": "m", "query": "q",
    ...                                    "documents": ["a"]})
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus

import httpx

from ...domain.endpoints.api_endpoint import ApiEndpoint
from ...domain.errors import RagDependencyError
from ..http.http_policy import HttpPolicy
from ..http.json_post import RetryingJsonPost, describe_malformed
from .bearer_key import BearerKey

# Retried like a 5xx whatever the key: a rate limit lifts by itself.
_ALWAYS_RETRIED = frozenset({HTTPStatus.TOO_MANY_REQUESTS})


class ApiEndpointClient:
    """POSTs JSON to one model endpoint and shapes every failure the same way.

    Args:
        client: The shared HTTP client every call goes through.
        endpoint: The route, its API format and its model.
        component: Names the dependency in every error: ``embedder``,
            ``reranker`` or ``llm`` (they become the workflow's error codes).
        key: Supplies the ``Authorization`` header of every attempt from a key
            source; None sends no key.
        policy: Timeout, retries and backoff of ``post``; ``post_once`` keeps the
            timeout and makes one attempt.

    Example:
        >>> endpoint_client = ApiEndpointClient(client, endpoint, component="llm",
        ...                                     key=None, policy=policy)
        >>> endpoint_client.described()
        'gpt-4o-mini at https://gateway/v1/chat/completions'
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        endpoint: ApiEndpoint,
        *,
        component: str,
        key: BearerKey | None,
        policy: HttpPolicy,
    ) -> None:
        self._endpoint, self._component, self._key = endpoint, component, key
        one_attempt = HttpPolicy(timeout_s=policy.timeout_s, retries=0, backoff_s=0.0)
        self._retrying = self._poster(client, policy)
        self._once = self._poster(client, one_attempt)

    def _poster(
        self, client: httpx.AsyncClient, policy: HttpPolicy
    ) -> RetryingJsonPost:
        # Both posters retry (or, with zero retries, report) the same statuses:
        # a 401 only with a key source, which may give another key.
        if self._key is None:
            return RetryingJsonPost(
                client, policy, headers=_no_key, retry_statuses=_ALWAYS_RETRIED
            )
        return RetryingJsonPost(
            client,
            policy,
            headers=self._key.headers,
            retry_statuses=_ALWAYS_RETRIED | {HTTPStatus.UNAUTHORIZED},
            on_retry_status=self._tell_refused_key,
            note_on_status=self._where_a_refused_key_comes_from,
        )

    @property
    def endpoint(self) -> ApiEndpoint:
        """The endpoint every call goes to.

        Returns:
            The route, its API format and its model.
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

    def _tell_refused_key(self, status: int, sent: Mapping[str, str]) -> None:
        # Only reached with a key source: a 401 says the key sent was refused.
        if status == HTTPStatus.UNAUTHORIZED and self._key is not None:
            self._key.refused_in(sent)

    def _where_a_refused_key_comes_from(self, status: int) -> str:
        # Only reached with a key source; the error of a 401 says what to do.
        if status != HTTPStatus.UNAUTHORIZED or self._key is None:
            return ""
        return f" (the key was refused: {self._key.described()})"


async def _no_key() -> dict[str, str]:
    # The headers of an endpoint called with no key.
    return {}
