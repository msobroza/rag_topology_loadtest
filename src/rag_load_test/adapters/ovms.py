"""HTTP adapters for OpenVINO Model Server (the ``ovms`` backends).

OVMS exposes OpenAI-compatible ``/v3/embeddings`` and Cohere-compatible
``/v3/rerank``; their paths and JSON shapes come from :mod:`.v3_contract`.
On Domino each OVMS app sits behind a proxy that expects the run's bearer
token, hence :class:`BearerToken`. Calls follow one :class:`HttpPolicy`: they
time out after ``timeout_s`` and are retried ``retries`` times on transport
errors, 5xx and 401, waiting ``backoff_s`` (doubled each time) in between.

Example::

    policy = HttpPolicy(timeout_s=30.0, retries=2, backoff_s=0.2)
    token = BearerToken(client, "", token_url=url, ttl_s=240.0, policy=policy)
    reranker = OvmsReranker(client, "http://localhost:8001", "bge-reranker-base",
                            token=token, policy=policy)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from http import HTTPStatus
from typing import NamedTuple

import httpx
from pydantic import ValidationError

from ..domain.errors import RagDependencyError, describe_error
from ..domain.passages import ScoredPassage
from ..domain.ports import EmbedderPort, RerankerPort
from .http_policy import HttpPolicy
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
    """``Authorization`` header: nothing, a static token, or Domino's run token,
    fetched from ``token_url`` and reused for ``ttl_s`` seconds.

    Example::

        token = BearerToken(client, "domino", token_url=url, ttl_s=240, policy=policy)
        await token.headers()   # {"Authorization": "Bearer <run token>"}
        # auth "abc" -> "Bearer abc"; auth "" -> {}
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
        if not self._auth:
            return {}
        token = await self._domino_token() if self._auth == "domino" else self._auth
        return {"Authorization": f"Bearer {token}"}

    def invalidate(self) -> None:
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


class _RetryableFailure(NamedTuple):
    # An attempt that failed in a way a retry may fix, and how to report it.
    detail: str
    kind: str


class OvmsEndpoint:
    """Shared plumbing for one OVMS servable: URLs, auth, retries, readiness."""

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

    async def ready(self) -> tuple[bool, str]:
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
        """POST JSON; retry on transport errors, 5xx and 401 (token dropped first).

        Args:
            path: The route under the servable's base URL.
            request: The body, sent in its wire format.

        Returns:
            The decoded JSON body of the first successful answer.

        Raises:
            RagDependencyError: When every attempt failed in a way a retry may
                fix; the error carries the last attempt's detail and kind.
        """
        url, payload = self._base_url + path, request.to_wire()
        failure = _RetryableFailure("no attempt made", "unavailable")
        for attempt in range(self._policy.retries + 1):
            if attempt:
                await asyncio.sleep(self._policy.backoff_before(attempt))
            outcome = await self._attempt(url, payload)
            if not isinstance(outcome, _RetryableFailure):
                return outcome
            failure = outcome
        raise RagDependencyError(
            self.component, failure.detail, target=url, kind=failure.kind
        )

    async def _attempt(self, url: str, payload: object) -> object:
        """POST once and sort the answer into a body, a retry or an error.

        Args:
            url: Where to post.
            payload: The JSON body.

        Returns:
            The decoded JSON body on success, or a ``_RetryableFailure`` when a
            retry may fix what went wrong.

        Raises:
            RagDependencyError: On a 4xx other than 401: retrying cannot fix a
                bad request.
        """
        try:
            headers = await self._token.headers()
            response = await self._client.post(
                url, json=payload, headers=headers, timeout=self._policy.timeout_s
            )
        except httpx.TransportError as exc:
            timed_out = isinstance(exc, httpx.TimeoutException)
            kind = "timeout" if timed_out else "unavailable"
            return _RetryableFailure(describe_error(exc), kind)
        status = response.status_code
        if status < HTTPStatus.BAD_REQUEST:
            return response.json()
        detail = f"HTTP {status}: {response.text[:200]}"
        if status == HTTPStatus.UNAUTHORIZED:
            self._token.invalidate()  # stale Domino token: refetch on retry
            return _RetryableFailure(detail, "unavailable")
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
            return _RetryableFailure(detail, "unavailable")
        # Any other 4xx: retrying cannot fix a bad request.
        raise RagDependencyError(self.component, detail, target=url)

    def _malformed(self, exc: Exception, path: str) -> RagDependencyError:
        return RagDependencyError(
            self.component,
            f"malformed response: {_describe_malformed(exc)}",
            target=self._base_url + path,
        )


class OvmsEmbedder(OvmsEndpoint, EmbedderPort):
    """EmbedderPort over ``POST /v3/embeddings`` (rows re-ordered by ``index``),
    ``batch_size`` texts per request."""

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
    """RerankerPort over ``POST /v3/rerank`` (``results[].index`` -> input order)."""

    component = "reranker"

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        if not passages:
            return []
        request = rerank_request_for(self.model, query, passages, top_n)
        body = await self._post(RERANK_PATH, request)
        try:
            response = RerankResponse.model_validate(body)
            return passages_from_rerank(response, passages, top_n)
        except (ValidationError, IndexError) as exc:  # IndexError: index out of range
            raise self._malformed(exc, RERANK_PATH) from exc


def _describe_malformed(exc: Exception) -> str:
    """One line naming what broke the shape, not the whole (maybe huge) body.

    Example::

        _describe_malformed(exc)  # "ValidationError for RerankResponse: results: ..."
    """
    if not isinstance(exc, ValidationError):
        return describe_error(exc)
    problems = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<body>'}: {error['msg']}"
        for error in exc.errors()
    )
    return f"ValidationError for {exc.title}: {problems}"
