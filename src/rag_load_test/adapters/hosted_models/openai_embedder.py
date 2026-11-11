"""The OpenAI embeddings API (the ``openai`` embedder backend).

Any OpenAI-compatible server works, a Domino AI Gateway included: a blank key or
base URL leaves the SDK to read ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``. The SDK
applies the :class:`HttpPolicy` timeout and retries itself; ``openai`` is
imported only when a client is built or a call fails, so importing this module
stays cheap.

Example:
    >>> embedder = OpenAIEmbedder.open("text-embedding-3-small", api_key="",
    ...                                base_url="", batch_size=64, policy=policy)
    >>> vectors = await embedder.embed(["vacation policy", "expense report"])
    >>> await embedder.aclose()
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...domain.errors import RagDependencyError
from ...domain.ports import EmbedderPort
from ..shared.http_policy import HttpPolicy

COMPONENT = "embedder"


class OpenAIEmbedder(EmbedderPort):
    """EmbedderPort on the OpenAI embeddings API (``AsyncOpenAI``).

    Args:
        client: An ``openai.AsyncOpenAI`` client, or a stand-in with the same
            ``embeddings.create`` and ``close``.
        model: The embedding model name sent with every request.
        batch_size: Texts per request.
    """

    def __init__(self, client: Any, model: str, *, batch_size: int) -> None:
        self._client, self._model, self._batch_size = client, model, batch_size
        self.model_name = f"openai:{model}"

    @classmethod
    def open(
        cls,
        model: str,
        *,
        api_key: str,
        base_url: str,
        batch_size: int,
        policy: HttpPolicy,
    ) -> OpenAIEmbedder:
        """Build the embedder on a fresh ``AsyncOpenAI`` client.

        Args:
            model: The embedding model name.
            api_key: The API key; "" leaves the SDK to read ``OPENAI_API_KEY``.
            base_url: The API base URL; "" leaves the SDK to its default.
            batch_size: Texts per request.
            policy: Timeout and retries of every request (the SDK retries
                itself, with its own backoff).

        Returns:
            The embedder. The SDK raises ``openai.OpenAIError`` here when no key
            is given and ``OPENAI_API_KEY`` is unset.
        """
        import openai

        client = openai.AsyncOpenAI(
            api_key=api_key or None,
            base_url=base_url or None,
            timeout=policy.timeout_s,
            max_retries=policy.retries,
        )
        return cls(client, model, batch_size=batch_size)

    async def aclose(self) -> None:
        """Close the SDK's HTTP connections."""
        await self._client.close()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts, ``batch_size`` per request.

        Args:
            texts: The texts to embed; none sends no request.

        Returns:
            One vector per text, in the same order.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def ready(self) -> tuple[bool, str]:
        """Report ready without calling the API (a probe would be billed).

        Returns:
            ``True`` and the model name.
        """
        return True, self.model_name

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        # One request; SDK errors become RagDependencyError (timeouts kind
        # "timeout"), and rows come back in input order.
        import openai

        try:
            response = await self._client.embeddings.create(
                model=self._model, input=batch
            )
        except openai.APITimeoutError as exc:
            raise RagDependencyError(
                COMPONENT, str(exc), target=self._model, kind="timeout"
            ) from exc
        except openai.APIError as exc:
            raise RagDependencyError(COMPONENT, str(exc), target=self._model) from exc
        return self._vectors_in_order(response.data, len(batch))

    def _vectors_in_order(self, rows: Sequence[Any], sent: int) -> list[list[float]]:
        # Rows may arrive in any order; each must name a distinct text sent.
        rows = sorted(rows, key=lambda row: row.index)
        indexes = [row.index for row in rows]
        if indexes != list(range(sent)):
            detail = (
                f"malformed response: row indexes {indexes} for {sent} texts sent "
                f"(expected 0..{sent - 1}, once each)"
            )
            raise RagDependencyError(COMPONENT, detail, target=self._model)
        return [list(row.embedding) for row in rows]
