"""HTTP adapters for OpenVINO Model Server (the ``ovms`` backends).

OVMS exposes OpenAI-compatible ``/v3/embeddings`` and Cohere-compatible
``/v3/rerank``; their paths and JSON shapes come from :mod:`.v3_contract`.
On Domino each OVMS app sits behind a proxy that expects the run's bearer
token, hence :class:`BearerToken`. Calls time out after
``TIMEOUT_S`` and are retried ``RETRIES`` times on transport errors and 5xx.

Example::

    reranker = OvmsReranker(client, "http://localhost:8001", "bge-reranker-base",
                            token=BearerToken(client, ""))
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import httpx
from pydantic import ValidationError

from .models import RagDependencyError, ScoredPassage, describe_error
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

TIMEOUT_S = 30.0
RETRIES = 2
DOMINO_TOKEN_URL = "http://localhost:8899/access-token"
DOMINO_TOKEN_TTL_S = 240.0  # Domino run tokens expire after ~5 minutes


class BearerToken:
    """``Authorization`` header: nothing, a static token, or Domino's cached run token.

    Example::

        await BearerToken(client, "abc").headers()      # Authorization: Bearer abc
        await BearerToken(client, "domino").headers()   # fetched from DOMINO_TOKEN_URL
        await BearerToken(client, "").headers()         # {}
    """

    def __init__(self, client: httpx.AsyncClient, auth: str) -> None:
        self._client, self._auth = client, auth
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
        if self._cached and time.monotonic() - self._fetched_at < DOMINO_TOKEN_TTL_S:
            return self._cached
        try:
            response = await self._client.get(DOMINO_TOKEN_URL, timeout=TIMEOUT_S)
        except httpx.TransportError as exc:
            raise RagDependencyError(
                "domino_access_token", describe_error(exc), target=DOMINO_TOKEN_URL
            ) from exc
        token = response.text.strip()
        if response.status_code != 200 or not token:
            detail = f"HTTP {response.status_code}: {response.text[:200]!r}"
            raise RagDependencyError(
                "domino_access_token", detail, target=DOMINO_TOKEN_URL
            )
        self._cached, self._fetched_at = token, time.monotonic()
        return token


class OvmsEndpoint:
    """Shared plumbing for one OVMS servable: URLs, auth, retries, readiness."""

    component = "ovms"
    backoff_s = 0.2  # doubled per attempt; tests set it to 0

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        *,
        token: BearerToken,
    ) -> None:
        self._client, self._token = client, token
        self._base_url, self.model = base_url.rstrip("/"), model
        self.model_name = f"ovms:{model}"

    async def ready(self) -> tuple[bool, str]:
        url = self._base_url + model_probe_path(self.model)
        try:
            headers = await self._token.headers()
            response = await self._client.get(url, headers=headers, timeout=TIMEOUT_S)
        except (httpx.TransportError, RagDependencyError) as exc:
            return False, describe_error(exc)
        return response.status_code == 200, f"{url} -> {response.status_code}"

    async def _post(self, path: str, request: V3Request) -> object:
        """POST JSON; retry on transport errors, 5xx and 401 (token dropped first)."""
        url, payload = self._base_url + path, request.to_wire()
        detail, kind = "no attempt made", "unavailable"
        for attempt in range(RETRIES + 1):
            if attempt:
                await asyncio.sleep(self.backoff_s * 2 ** (attempt - 1))
            try:
                headers = await self._token.headers()
                response = await self._client.post(
                    url, json=payload, headers=headers, timeout=TIMEOUT_S
                )
            except httpx.TransportError as exc:
                detail = describe_error(exc)
                kind = (
                    "timeout"
                    if isinstance(exc, httpx.TimeoutException)
                    else "unavailable"
                )
                continue
            status = response.status_code
            if status == 401:
                self._token.invalidate()  # stale Domino token: refetch on retry
            if status == 401 or status >= 500:
                detail, kind = f"HTTP {status}: {response.text[:200]}", "unavailable"
                continue
            if status >= 400:  # retrying cannot fix a bad request
                raise RagDependencyError(
                    self.component, f"HTTP {status}: {response.text[:200]}", target=url
                )
            return response.json()
        raise RagDependencyError(self.component, detail, target=url, kind=kind)

    def _malformed(self, exc: Exception, path: str) -> RagDependencyError:
        return RagDependencyError(
            self.component,
            f"malformed response: {_describe_malformed(exc)}",
            target=self._base_url + path,
        )


class OvmsEmbedder(OvmsEndpoint):
    """EmbedderPort over ``POST /v3/embeddings`` (rows re-ordered by ``index``)."""

    component = "embedder"
    batch_size = 64

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


class OvmsReranker(OvmsEndpoint):
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
