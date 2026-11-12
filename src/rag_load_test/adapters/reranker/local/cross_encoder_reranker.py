"""The ``local`` reranker: a cross-encoder model in this process.

Inside the pool's worker, each call runs in the model's ``NativeThreadScope``,
so ``workers x threads`` stays within the CPU budget. Heavy imports live in
the methods that need them, so importing this module stays cheap.

Example:
    >>> reranker = CrossEncoderReranker.from_pretrained(
    ...     model, pool, batch_size=32, threads=2)
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Any

from ....domain.passages import ScoredPassage, top_by_score
from ...local_models.local_model import LocalModel
from ...local_models.model_pool import run_blocking
from ...local_models.native_threads import NativeThreadScope
from ..base_reranker import BaseReranker


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
