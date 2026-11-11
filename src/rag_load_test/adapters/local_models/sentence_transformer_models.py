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

from ...domain.passages import ScoredPassage, top_by_score
from ...domain.ports import BaseEmbedder, BaseReranker
from .local_model import LocalModel
from .native_threads import NativeThreadScope

T = TypeVar("T")


def model_executor(
    workers: int, on_thread_start: Callable[[], None] | None = None
) -> ThreadPoolExecutor:
    """Build the shared pool for sentence-transformers inference.

    ``workers`` calls run at once (from the thread budget), the rest queue.

    Args:
        workers: Inference calls that run at once.
        on_thread_start: Runs once in each new worker thread
            (``settle_torch_thread`` for real models); ``None`` runs nothing.

    Returns:
        The executor, its threads named ``rag-model``.

    Example:
        >>> pool = model_executor(4, on_thread_start=settle_torch_thread)
    """
    return ThreadPoolExecutor(
        workers, thread_name_prefix="rag-model", initializer=on_thread_start
    )


async def run_blocking(executor: Executor, fn: Callable[..., T], *args: Any) -> T:
    """Await ``fn(*args)`` on ``executor`` (``functools.partial`` for kwargs).

    Args:
        executor: The pool the blocking call runs on.
        fn: The blocking call.
        *args: Positional arguments passed to ``fn``.

    Returns:
        What ``fn`` returns.
    """
    return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)


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


class CrossEncoderReranker(BaseReranker):
    """BaseReranker on a sentence-transformers ``CrossEncoder`` (scores in 0..1).

    Args:
        model: A ``CrossEncoder``, or a fake with the same ``predict``.
        executor: The shared inference pool the predict calls run on.
        model_name: The name ``ready`` reports.
        batch_size: Query-passage pairs per ``predict`` batch.
        scope: Caps the OpenMP threads of each predict call.

    Example:
        >>> model = local_model("org/reranker", runtime="torch", models_dir="models")
        >>> reranker = CrossEncoderReranker.from_pretrained(
        ...     model, pool, batch_size=32, threads=2
        ... )
        >>> best = await reranker.rerank("vacation policy", candidates, top_n=5)
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
    ) -> CrossEncoderReranker:
        """Load a ``CrossEncoder`` on CPU, then build its thread scope.

        Args:
            model: The model to load, and the runtime it loads under.
            executor: The shared inference pool the predict calls run on.
            batch_size: Query-passage pairs per ``predict`` batch.
            threads: Threads one predict call may use.

        Returns:
            The reranker on the loaded model.
        """
        from sentence_transformers import CrossEncoder  # pulls in torch

        loaded = CrossEncoder(
            model.source, device="cpu", **model.load_arguments(threads)
        )
        # After the model: the scope only sees native libraries already loaded.
        scope = NativeThreadScope(threads)
        return cls(
            loaded, executor, model_name=model.name, batch_size=batch_size, scope=scope
        )

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        """Score each (query, passage) pair on the pool and keep the best.

        Args:
            query: The question the passages are scored against.
            passages: The retrieval candidates; none returns ``[]`` without a model
                call.
            top_n: How many passages to keep.

        Returns:
            At most ``top_n`` passages, best first, with the cross-encoder score.
        """
        if not passages:
            return []
        pairs = [(query, p.text) for p in passages]
        predict = functools.partial(
            self._model.predict, pairs, batch_size=self._batch_size
        )
        scores = await run_blocking(self._executor, self.scope.run, predict)
        return top_by_score(passages, [float(s) for s in scores], top_n)

    async def ready(self) -> tuple[bool, str]:
        """Report ready: the model was loaded when the object was built.

        Returns:
            ``True`` and the model name.
        """
        return True, self.model_name


class NoReranker(BaseReranker):
    """BaseReranker that keeps retrieval order (``RAG_RERANK_BACKEND=none``)."""

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
        """Report ready; there is no model.

        Returns:
            ``True`` and ``"none"``.
        """
        return True, "none"
