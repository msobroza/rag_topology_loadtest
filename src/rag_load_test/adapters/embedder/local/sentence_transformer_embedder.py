"""The ``local`` embedder: a sentence-transformers model in this process.

Inside the pool's worker, each call runs in the model's ``NativeThreadScope``,
so ``workers x threads`` stays within the CPU budget. Heavy imports live in
the methods that need them, so importing this module stays cheap.

Example:
    >>> embedder = SentenceTransformerEmbedder.from_pretrained(
    ...     model, pool, batch_size=64, threads=2)
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Any

from ...local_models.local_model import LocalModel
from ...local_models.model_pool import run_blocking
from ...local_models.native_threads import NativeThreadScope
from ..base_embedder import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """BaseEmbedder on a ``SentenceTransformer``; inject a fake ``model`` in tests.

    Args:
        model: A ``SentenceTransformer``, or a fake with the same ``encode``.
        executor: The shared inference pool the encode calls run on.
        model_name: The name ``ready`` reports.
        batch_size: Texts per ``encode`` batch.
        scope: Caps the OpenMP threads of each encode call.

    Example:
        >>> model = local_model("org/embedder", runtime="torch", models_dir="models")
        >>> embedder = SentenceTransformerEmbedder.from_pretrained(
        ...     model, pool, batch_size=64, threads=2
        ... )
        >>> [vector] = await embedder.embed(["vacation policy"])
    """

    def __init__(
        self,
        model: Any,
        executor: Executor,
        *,
        model_name: str,
        batch_size: int,
        scope: NativeThreadScope,
    ) -> None:
        self._model, self._executor, self._batch_size = model, executor, batch_size
        self.model_name, self.scope = model_name, scope

    @classmethod
    def from_pretrained(
        cls, model: LocalModel, executor: Executor, *, batch_size: int, threads: int
    ) -> SentenceTransformerEmbedder:
        """Load a ``SentenceTransformer`` on CPU, then build its thread scope.

        Args:
            model: The model to load, and the runtime it loads under.
            executor: The shared inference pool the encode calls run on.
            batch_size: Texts per ``encode`` batch.
            threads: Threads one encode call may use.

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
            loaded, executor, model_name=model.name, batch_size=batch_size, scope=scope
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts on the pool, inside the model's thread scope.

        Args:
            texts: The texts to embed; none returns ``[]`` without a model call.

        Returns:
            One unit-norm vector per text, in the same order.
        """
        if not texts:
            return []
        return await run_blocking(
            self._executor, self.scope.run, self._encode, list(texts)
        )

    async def ready(self) -> tuple[bool, str]:
        """Report ready: the model was loaded when the object was built.

        Returns:
            ``True`` and the model name.
        """
        return True, self.model_name

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Unit-norm vectors, so cosine similarity is a plain dot product.
        matrix = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True
        )
        return matrix.tolist()
