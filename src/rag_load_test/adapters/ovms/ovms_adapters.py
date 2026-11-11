"""HTTP adapters for OpenVINO Model Server (the ``ovms`` backends).

OVMS exposes OpenAI-compatible ``/v3/embeddings`` and Cohere-compatible
``/v3/rerank``; their paths and JSON shapes come from :mod:`.v3_contract`.
On Domino each OVMS app sits behind a proxy that expects the run's bearer
token, hence :class:`BearerToken`. Calls follow one :class:`HttpPolicy`: they
time out after ``timeout_s`` and are retried ``retries`` times on transport
errors, 5xx and 401, waiting ``backoff_s`` (doubled each time) in between.

Example:
    >>> policy = HttpPolicy(timeout_s=30.0, retries=2, backoff_s=0.2)
    >>> token = BearerToken(client, "", token_url=url, ttl_s=240.0, policy=policy)
    >>> reranker = OvmsReranker(client, "http://localhost:8001", "reranker-alias",
    ...                         token=token, policy=policy)
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from http import HTTPStatus

import httpx
from pydantic import ValidationError

from ...domain.errors import RagDependencyError, describe_error
from ...domain.passages import ScoredPassage
from ...domain.ports import EmbedderPort, RerankerPort
from ..shared.http_policy import HttpPolicy
from ..shared.json_post import RetryingJsonPost, describe_malformed
from .v3_contract import (
    EMBEDDINGS_PATH,
    RERANK_PATH,
    EmbeddingsRequest,
    EmbeddingsResponse,
    RerankResponse,
    V3Request,
    model_probe_path,
    passages_from_rerank,
    rerank_request_for,
    vectors_from_embeddings,
)


class BearerToken:
    """The ``Authorization`` header for OVMS calls.

    It is nothing, a static token, or Domino's run token, fetched from
    ``token_url`` and reused for ``ttl_s`` seconds.

    Args:
        client: The shared HTTP client the run token is fetched with.
        auth: "" sends no header; "domino" sends the run token; any other value
            is sent as a static token.
        token_url: Where Domino's run token is fetched (``auth="domino"`` only).
        ttl_s: Seconds a fetched run token is reused before it is fetched again.
        policy: Only its ``timeout_s`` is used: a token fetch is tried once, and a
            failed fetch fails the model call, which retries.

    Example:
        >>> token = BearerToken(client, "domino", token_url=url, ttl_s=240,
        ...                     policy=policy)
        >>> await token.headers()
        {'Authorization': 'Bearer <run token>'}
        >>> # auth "abc" -> "Bearer abc"; auth "" -> {}
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        auth: str,
        *,
        token_url: str,
        ttl_s: float,
        policy: HttpPolicy,
    ) -> None:
        self._client, self._auth = client, auth
        # One attempt only: a failed fetch fails the model call, which retries.
        self._token_url, self._ttl_s = token_url, ttl_s
        self._timeout_s = policy.timeout_s
        self._cached: str | None = None
        self._fetched_at = 0.0

    async def headers(self) -> dict[str, str]:
        """Return the headers to send, fetching the run token when it is stale.

        Returns:
            ``{"Authorization": "Bearer <token>"}``, or ``{}`` when ``auth`` is "".
        """
        if not self._auth:
            return {}
        token = await self._domino_token() if self._auth == "domino" else self._auth
        return {"Authorization": f"Bearer {token}"}

    def invalidate(self) -> None:
        """Drop the cached run token, so the next ``headers`` fetches a new one."""
        self._cached = None

    async def _domino_token(self) -> str:
        if self._cached and time.monotonic() - self._fetched_at < self._ttl_s:
            return self._cached
        url = self._token_url
        try:
            response = await self._client.get(url, timeout=self._timeout_s)
        except httpx.TransportError as exc:
            raise RagDependencyError(
                "domino_access_token", describe_error(exc), target=url
            ) from exc
        token = response.text.strip()
        if response.status_code != HTTPStatus.OK or not token:
            detail = f"HTTP {response.status_code}: {response.text[:200]!r}"
            raise RagDependencyError("domino_access_token", detail, target=url)
        self._cached, self._fetched_at = token, time.monotonic()
        return token


# Retried like a 5xx: the Domino proxy's 401 for an expired run token.
_OVMS_RETRY_STATUSES = frozenset({HTTPStatus.UNAUTHORIZED})


class OvmsEndpoint:
    """Shared plumbing for one OVMS servable: URLs, auth, retries, readiness.

    Args:
        client: The shared HTTP client every call goes through.
        base_url: The OVMS server's URL; a trailing slash is dropped.
        model: The servable's name in OVMS; ``model_name`` is ``ovms:<model>``.
        token: Supplies the ``Authorization`` header of every call.
        policy: Timeout, retries and backoff of every call.
    """

    component = "ovms"

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        *,
        token: BearerToken,
        policy: HttpPolicy,
    ) -> None:
        self._client, self._token, self._policy = client, token, policy
        self._base_url, self.model = base_url.rstrip("/"), model
        self.model_name = f"ovms:{model}"
        self._poster = RetryingJsonPost(
            client,
            policy,
            headers=token.headers,
            retry_statuses=_OVMS_RETRY_STATUSES,
            on_retry_status=self._drop_stale_token,
        )

    async def ready(self) -> tuple[bool, str]:
        """Probe the servable's model endpoint once, without retries.

        Returns:
            Whether the probe answered 200, and ``"<url> -> <status>"``; on a
            transport or token error, ``False`` and the error.
        """
        url = self._base_url + model_probe_path(self.model)
        try:
            headers = await self._token.headers()
            response = await self._client.get(
                url, headers=headers, timeout=self._policy.timeout_s
            )
        except (httpx.TransportError, RagDependencyError) as exc:
            return False, describe_error(exc)
        ok = response.status_code == HTTPStatus.OK
        return ok, f"{url} -> {response.status_code}"

    async def _post(self, path: str, request: V3Request) -> object:
        # POST under the policy; 401 is retried too, after the token is dropped.
        url = self._base_url + path
        return await self._poster.post(self.component, url, request.to_wire())

    def _drop_stale_token(self, status: int) -> None:
        # A 401 from the Domino proxy means the cached run token expired.
        if status == HTTPStatus.UNAUTHORIZED:
            self._token.invalidate()

    def _malformed(self, exc: Exception, path: str) -> RagDependencyError:
        return RagDependencyError(
            self.component,
            f"malformed response: {describe_malformed(exc)}",
            target=self._base_url + path,
        )


class OvmsEmbedder(OvmsEndpoint, EmbedderPort):
    """EmbedderPort over ``POST /v3/embeddings`` (rows re-ordered by ``index``).

    Args:
        client: The shared HTTP client every call goes through.
        base_url: The OVMS server's URL; a trailing slash is dropped.
        model: The servable's name in OVMS.
        token: Supplies the ``Authorization`` header of every call.
        policy: Timeout, retries and backoff of every call.
        batch_size: Texts per request.
    """

    component = "embedder"

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        *,
        token: BearerToken,
        policy: HttpPolicy,
        batch_size: int,
    ) -> None:
        super().__init__(client, base_url, model, token=token, policy=policy)
        self.batch_size = batch_size

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts, ``batch_size`` per request.

        Args:
            texts: The texts to embed; none sends no request.

        Returns:
            One vector per text, in the same order.

        Raises:
            RagDependencyError: If a response does not have the embeddings shape.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            # ``input`` stays a list even for one text: the wire format OVMS is sent.
            request = EmbeddingsRequest(model=self.model, input=batch)
            body = await self._post(EMBEDDINGS_PATH, request)
            try:
                response = EmbeddingsResponse.model_validate(body)
            except ValidationError as exc:
                raise self._malformed(exc, EMBEDDINGS_PATH) from exc
            vectors.extend(vectors_from_embeddings(response))
        return vectors


class OvmsReranker(OvmsEndpoint, RerankerPort):
    """RerankerPort over ``POST /v3/rerank`` (``results[].index`` -> input order).

    Built with the same arguments as ``OvmsEndpoint``.
    """

    component = "reranker"

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        """Re-score the candidates on OVMS and keep the best.

        Args:
            query: The question the passages are scored against.
            passages: The retrieval candidates; none sends no request.
            top_n: How many passages to keep.

        Returns:
            At most ``top_n`` passages, best first, with the OVMS score.

        Raises:
            RagDependencyError: If the response does not have the rerank shape, or
                names an index outside the candidates.
        """
        if not passages:
            return []
        request = rerank_request_for(self.model, query, passages, top_n)
        body = await self._post(RERANK_PATH, request)
        try:
            response = RerankResponse.model_validate(body)
            return passages_from_rerank(response, passages, top_n)
        except (ValidationError, IndexError) as exc:  # IndexError: index out of range
            raise self._malformed(exc, RERANK_PATH) from exc
