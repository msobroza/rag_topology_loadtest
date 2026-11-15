"""The ``local`` embedder: a sentence-transformers model in this process.

Inside the pool's worker, each call runs in the model's ``NativeThreadScope``,
so ``workers x threads`` stays within the CPU budget. Heavy imports live in
the methods that need them, so importing this module stays cheap.

Example:
    >>> embedder = SentenceTransformerEmbedder.from_pretrained(
    ...     model, pool, batch_size=64, threads=2, query_prefix="query: ")
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from typing import Any

from ....infrastructure.local_models.local_model import LocalModel
from ....infrastructure.local_models.model_pool import run_blocking
from ....infrastructure.local_models.native_threads import NativeThreadScope
from ..base_embedder import BaseEmbedder


@dataclass(frozen=True)
class EncodeOptions:
    """How each ``encode`` call is made: its batch size and its normalisation.

    Example:
        >>> EncodeOptions(batch_size=64, normalize_embeddings=False)
    """

    batch_size: int  # texts per encode batch
    # Scale each vector to unit length. Off is safe: every store compares
    # vectors by cosine, which ignores their length.
    normalize_embeddings: bool


class SentenceTransformerEmbedder(BaseEmbedder):
    """BaseEmbedder on a ``SentenceTransformer``; inject a fake ``model`` in tests.

    Args:
        model: A ``SentenceTransformer``, or a fake with the same ``encode``.
        executor: The shared inference pool the encode calls run on.
        model_name: The name ``ready`` reports.
        scope: Caps the OpenMP threads of each encode call.
        query_prefix: Put in front of a query, never of a passage; "" for a
            model that takes none.
        encode_options: The batch size and normalisation of each encode call.

    Example:
        >>> model = local_model("org/embedder", runtime="torch", models_dir="models")
        >>> embedder = SentenceTransformerEmbedder.from_pretrained(
        ...     model, pool, batch_size=64, threads=2, query_prefix="query: "
        ... )
        >>> vector = await embedder.embed_query("vacation policy")
    """

    def __init__(
        self,
        model: Any,
        executor: Executor,
        *,
        model_name: str,
        scope: NativeThreadScope,
        query_prefix: str,
        encode_options: EncodeOptions,
    ) -> None:
        self._model, self._executor = model, executor
        self.model_name, self.scope = model_name, scope
        self._query_prefix, self._encode_options = query_prefix, encode_options

    @classmethod
    def from_pretrained(
        cls,
        model: LocalModel,
        executor: Executor,
        *,
        batch_size: int,
        threads: int,
        query_prefix: str,
        normalize_embeddings: bool = False,
    ) -> SentenceTransformerEmbedder:
        """Load a ``SentenceTransformer`` on CPU, then build its thread scope.

        Args:
            model: The model to load, and the runtime it loads under.
            executor: The shared inference pool the encode calls run on.
            batch_size: Texts per ``encode`` batch.
            threads: Threads one encode call may use.
            query_prefix: Put in front of a query, never of a passage.
            normalize_embeddings: Scale each vector to unit length.

        Returns:
            The embedder on the loaded model.
        """
        from sentence_transformers import SentenceTransformer  # pulls in torch

        loaded = SentenceTransformer(
            model.source, device="cpu", **model.load_arguments(threads)
        )
        # After the model: the scope only sees native libraries already loaded.
        scope = NativeThreadScope(threads)
        return cls(
            loaded,
            executor,
            model_name=model.name,
            scope=scope,
            query_prefix=query_prefix,
            encode_options=EncodeOptions(batch_size, normalize_embeddings),
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts as they are, on the pool, inside the model's thread scope.

        Args:
            texts: The texts to embed; none returns ``[]`` without a model call.

        Returns:
            One vector per text, in the same order; unit length only when
            ``normalize_embeddings`` is set.
        """
        if not texts:
            return []
        return await self._run_encode(list(texts))

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query with the query prefix in front, as one encode call.

        Args:
            text: The query, as the user asked it.

        Returns:
            Its vector; unit length only when ``normalize_embeddings`` is set.

        Example:
            >>> vector = await embedder.embed_query("How many vacation days?")
        """
        [vector] = await self._run_encode([self._query_prefix + text])
        return vector

    def with_batch_size(self, batch_size: int) -> SentenceTransformerEmbedder:
        """Return this adapter at another batch size, on the same loaded model.

        Calibration times one loaded model under several batch sizes, and a load
        is the expensive part.

        Args:
            batch_size: Texts per ``encode`` batch.

        Returns:
            A new adapter sharing the model, the pool, the model name, the
            thread scope, the query prefix and the normalisation.

        Example:
            >>> smaller = adapter.with_batch_size(2)
        """
        return SentenceTransformerEmbedder(
            self._model,
            self._executor,
            model_name=self.model_name,
            scope=self.scope,
            query_prefix=self._query_prefix,
            encode_options=replace(self._encode_options, batch_size=batch_size),
        )

    async def ready(self) -> tuple[bool, str]:
        """Report ready: the model was loaded when the object was built.

        Returns:
            ``True`` and the model name.
        """
        return True, self.model_name

    async def _run_encode(self, texts: list[str]) -> list[list[float]]:
        return await run_blocking(self._executor, self.scope.run, self._encode, texts)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Normalising is optional: every store compares by cosine, which ignores
        # length. No progress bar: at the Apps' INFO level one is drawn on stderr
        # per call.
        options = self._encode_options
        matrix = self._model.encode(
            texts,
            batch_size=options.batch_size,
            normalize_embeddings=options.normalize_embeddings,
            show_progress_bar=False,
        )
        return matrix.tolist()
