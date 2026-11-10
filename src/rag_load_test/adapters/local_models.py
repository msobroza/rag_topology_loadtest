"""In-process sentence-transformers models: the embedder and the rerankers.

sentence-transformers is synchronous, so model calls go through ``run_blocking``
on one bounded ``ThreadPoolExecutor``: the event loop stays free and overload
turns into queueing. Inside the worker, each call runs in its model's
``NativeThreadScope``, so ``workers x threads`` stays within the CPU budget.
Heavy imports live inside the methods that need them so ``import rag_load_test``
stays cheap for the OVMS and fake apps.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, TypeVar

from ..domain.passages import ScoredPassage, top_by_score
from ..domain.ports import EmbedderPort, RerankerPort
from .native_threads import NativeThreadScope

T = TypeVar("T")


def model_executor(
    workers: int, on_thread_start: Callable[[], None] | None = None
) -> ThreadPoolExecutor:
    """The shared pool for sentence-transformers inference: ``workers`` calls run
    at once (from the thread budget), the rest queue. ``on_thread_start`` runs
    once in each new worker thread (``settle_torch_thread`` for real models).

    Example::

        pool = model_executor(4, on_thread_start=settle_torch_thread)
    """
    return ThreadPoolExecutor(
        workers, thread_name_prefix="rag-model", initializer=on_thread_start
    )


async def run_blocking(executor: Executor, fn: Callable[..., T], *args: Any) -> T:
    """Await ``fn(*args)`` on ``executor`` (``functools.partial`` for kwargs)."""
    return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)


class SentenceTransformerEmbedder(EmbedderPort):
    """EmbedderPort on a ``SentenceTransformer``; inject a fake ``model`` in tests.

    Example::

        embedder = SentenceTransformerEmbedder.from_pretrained(
            "BAAI/bge-small", pool, batch_size=64, threads=2
        )
        [vector] = await embedder.embed(["vacation policy"])
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
        cls, model_name: str, executor: Executor, *, batch_size: int, threads: int
    ) -> SentenceTransformerEmbedder:
        from sentence_transformers import SentenceTransformer  # pulls in torch

        model = SentenceTransformer(model_name, device="cpu")
        # After the model: the scope only sees native libraries already loaded.
        scope = NativeThreadScope(threads)
        return cls(
            model, executor, model_name=model_name, batch_size=batch_size, scope=scope
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await run_blocking(
            self._executor, self.scope.run, self._encode, list(texts)
        )

    async def ready(self) -> tuple[bool, str]:
        return True, self.model_name

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Unit-norm vectors, so cosine similarity is a plain dot product.
        matrix = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True
        )
        return matrix.tolist()


class CrossEncoderReranker(RerankerPort):
    """RerankerPort on a sentence-transformers ``CrossEncoder`` (scores in 0..1).

    Example::

        reranker = CrossEncoderReranker.from_pretrained(
            "BAAI/bge-reranker-base", pool, batch_size=32, threads=2
        )
        best = await reranker.rerank("vacation policy", candidates, top_n=5)
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
        cls, model_name: str, executor: Executor, *, batch_size: int, threads: int
    ) -> CrossEncoderReranker:
        from sentence_transformers import CrossEncoder  # pulls in torch

        model = CrossEncoder(model_name, device="cpu")
        # After the model: the scope only sees native libraries already loaded.
        scope = NativeThreadScope(threads)
        return cls(
            model, executor, model_name=model_name, batch_size=batch_size, scope=scope
        )

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        if not passages:
            return []
        pairs = [(query, p.text) for p in passages]
        predict = functools.partial(
            self._model.predict, pairs, batch_size=self._batch_size
        )
        scores = await run_blocking(self._executor, self.scope.run, predict)
        return top_by_score(passages, [float(s) for s in scores], top_n)

    async def ready(self) -> tuple[bool, str]:
        return True, self.model_name


class NoReranker(RerankerPort):
    """RerankerPort that keeps retrieval order (``RAG_RERANKER_BACKEND=none``)."""

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        """Keep the first passages, in retrieval order.

        Args:
            query: Unused: retrieval order does not depend on it.
            passages: The candidates, best retrieval score first.
            top_n: How many to keep.

        Returns:
            The first ``top_n`` passages, as given.
        """
        del query  # the port's signature; retrieval order needs no query
        return list(passages[:top_n])

    async def ready(self) -> tuple[bool, str]:
        return True, "none"
