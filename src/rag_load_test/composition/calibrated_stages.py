"""The CPU-bound stages calibration times: rerank and embed on an in-process model.

Each stage says how its adapter loads for a thread count, takes another batch
size and serves one request of the usual size, on the path a real request
takes; ``infrastructure/local_models/stage_calibration.py`` times them. The
factories pick the candidates from the settings with ``candidates``: a setting
above 0 is the one candidate, 0 searches.

Example:
    >>> reranker = await calibrate(
    ...     RerankStage(model, pool, settings),
    ...     threads=candidates(settings.rerank_threads, budget.reranker_threads),
    ...     batch_sizes=candidates(settings.rerank_batch_size, settings.top_k_retrieve),
    ...     limits=calibration_limits(settings),
    ... )
"""

from __future__ import annotations

from concurrent.futures import Executor

from ..adapters.embedder.local.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from ..adapters.reranker.local.cross_encoder_reranker import CrossEncoderReranker
from ..config.rag_settings import RagSettings
from ..domain.calibration import calibration_sample, halvings
from ..domain.passages import ScoredPassage
from ..infrastructure.local_models.local_model import LocalModel
from ..infrastructure.local_models.stage_calibration import (
    BaseCalibratedStage,
    CalibrationLimits,
)


def candidates(setting: int, top: int) -> tuple[int, ...]:
    """Return the candidates of one parameter: the setting if above 0, else a search.

    Args:
        setting: The parameter's setting; 0 = calibrated at start-up.
        top: The largest candidate of a search.

    Returns:
        ``(setting,)``, or ``halvings(top)`` when ``setting`` is 0.

    Example:
        >>> candidates(0, 20)
        (20, 10, 5, 2, 1)
    """
    return (setting,) if setting > 0 else halvings(top)


def calibration_limits(settings: RagSettings) -> CalibrationLimits:
    """Return the calibration budget and rounds of one stage, from the settings.

    Args:
        settings: ``calibration_budget_s`` and ``calibration_rounds``.

    Returns:
        The limits every stage is calibrated under.

    Example:
        >>> calibration_limits(RagSettings()).budget_s
        90.0
    """
    return CalibrationLimits(
        budget_s=settings.calibration_budget_s, rounds=settings.calibration_rounds
    )


class RerankStage(BaseCalibratedStage[CrossEncoderReranker]):
    """The rerank stage on the in-process cross-encoder.

    Its sample is a question and ``top_k_retrieve`` passages, kept to
    ``top_k_rerank``, as the workflow reranks them.

    Args:
        model: The model's files and runtime.
        pool: The shared inference pool.
        settings: ``top_k_retrieve`` and ``top_k_rerank``.

    Example:
        >>> stage = RerankStage(model, pool, settings)
        >>> reranker = await stage.loaded(threads=2)
    """

    stage = "rerank"

    def __init__(
        self, model: LocalModel, pool: Executor, settings: RagSettings
    ) -> None:
        self._model, self._pool = model, pool
        sample = calibration_sample(settings.top_k_retrieve)
        self._question, self._top_n = sample.question, settings.top_k_rerank
        self._passages = [
            ScoredPassage(id=f"calibration-{i}", text=text, retrieval_score=0.0)
            for i, text in enumerate(sample.passages)
        ]

    async def loaded(self, threads: int) -> CrossEncoderReranker:
        """Load the cross-encoder for ``threads`` threads.

        Args:
            threads: Native threads one predict call may use.

        Returns:
            The reranker, at batch size 1 until ``rebatched``.
        """
        return CrossEncoderReranker.from_pretrained(
            self._model, self._pool, batch_size=1, threads=threads
        )

    def rebatched(
        self, adapter: CrossEncoderReranker, batch_size: int
    ) -> CrossEncoderReranker:
        """Return ``adapter`` at ``batch_size`` passages per predict batch.

        Args:
            adapter: A loaded reranker.
            batch_size: Passages per predict batch.

        Returns:
            The reranker on the same model.
        """
        return adapter.with_batch_size(batch_size)

    async def serve_sample(self, adapter: CrossEncoderReranker) -> None:
        """Rerank the sample's passages, as the workflow reranks its candidates.

        Args:
            adapter: The reranker under trial.
        """
        await adapter.rerank(self._question, self._passages, top_n=self._top_n)


class EmbedStage(BaseCalibratedStage[SentenceTransformerEmbedder]):
    """The embed stage on the in-process sentence-transformers model.

    Its sample is one question with the query prefix, as the workflow embeds
    it; a query is one text, so its batch size is never searched.

    Args:
        model: The model's files and runtime.
        pool: The shared inference pool.
        settings: ``embedding_query_prefix``.

    Example:
        >>> stage = EmbedStage(model, pool, settings)
        >>> embedder = await stage.loaded(threads=2)
    """

    stage = "embed"

    def __init__(
        self, model: LocalModel, pool: Executor, settings: RagSettings
    ) -> None:
        self._model, self._pool = model, pool
        question = calibration_sample(1).question
        self._query = settings.embedding_query_prefix + question

    async def loaded(self, threads: int) -> SentenceTransformerEmbedder:
        """Load the embedding model for ``threads`` threads.

        Args:
            threads: Native threads one encode call may use.

        Returns:
            The embedder, at batch size 1 until ``rebatched``.
        """
        return SentenceTransformerEmbedder.from_pretrained(
            self._model, self._pool, batch_size=1, threads=threads
        )

    def rebatched(
        self, adapter: SentenceTransformerEmbedder, batch_size: int
    ) -> SentenceTransformerEmbedder:
        """Return ``adapter`` at ``batch_size`` texts per encode batch.

        Args:
            adapter: A loaded embedder.
            batch_size: Texts per encode batch.

        Returns:
            The embedder on the same model.
        """
        return adapter.with_batch_size(batch_size)

    async def serve_sample(self, adapter: SentenceTransformerEmbedder) -> None:
        """Embed the sample question, as the workflow embeds a question.

        Args:
            adapter: The embedder under trial.
        """
        await adapter.embed([self._query])
