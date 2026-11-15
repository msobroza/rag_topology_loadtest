"""The CPU-bound stages calibration times: rerank and embed on an in-process model.

Each stage says how its adapter loads for a thread count, takes another batch
size, serves one request of the usual size, on the path a real request takes,
and which ``RAG_*`` settings pin a tuning;
``infrastructure/local_models/stage_calibration.py`` times them. The
factories pick the candidates from the settings with ``candidates``: a setting
above 0 is the one candidate, 0 searches.

Example:
    >>> calibrated = await calibrate(
    ...     RerankStage(model, pool, settings),
    ...     threads=candidates(settings.rerank_threads, budget.reranker_threads),
    ...     batch_sizes=candidates(settings.rerank_batch_size, settings.top_k_retrieve),
    ...     limits=calibration_limits(settings),
    ... )
    >>> reranker = calibrated.adapter
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import TypeVar

from ..adapters.embedder.local.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from ..adapters.reranker.local.cross_encoder_reranker import CrossEncoderReranker
from ..config.rag_settings import RagSettings
from ..domain.calibration import (
    CalibrationReport,
    StageTuning,
    calibration_sample,
    halvings,
)
from ..domain.passages import ScoredPassage
from ..infrastructure.local_models.local_model import LocalModel
from ..infrastructure.local_models.stage_calibration import (
    BaseCalibratedStage,
    CalibrationLimits,
    calibrate,
)

T = TypeVar("T")


@dataclass(frozen=True)
class ComponentCalibration:
    """One component's calibration on this machine, and the settings that pin it.

    Example:
        >>> calibration = await composition.calibration("reranker")
        >>> calibration.pinned
        {'RAG_RERANK_THREADS': 2, 'RAG_RERANK_BATCH_SIZE': 2}
    """

    report: CalibrationReport
    cpus: float  # the CPUs the thread budget counted here
    runtime: str  # what ran the model: openvino or torch
    pinned: dict[str, int]  # the variables that pin the choice, by name


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


async def component_calibration(
    stage: BaseCalibratedStage[T],
    *,
    threads: Sequence[int],
    batch_sizes: Sequence[int],
    settings: RagSettings,
    cpus: float,
) -> ComponentCalibration:
    """Calibrate ``stage`` over the candidates given, for ``rag-calibrate``.

    The adapter calibration returns is dropped here, so its model is released
    before the next component loads.

    Args:
        stage: The stage to time.
        threads: Thread counts to try, the budget's share first.
        batch_sizes: Batch sizes to try, the whole request first.
        settings: The calibration budget and rounds, and the runtime.
        cpus: The CPUs the budget counted, for the report.

    Returns:
        The report and the settings that pin its choice.

    Example:
        >>> await component_calibration(stage, threads=(2, 1), batch_sizes=(20, 10),
        ...                             settings=settings, cpus=2.0)
    """
    calibrated = await calibrate(
        stage,
        threads=threads,
        batch_sizes=batch_sizes,
        limits=calibration_limits(settings),
    )
    report = calibrated.report
    return ComponentCalibration(
        report,
        cpus=cpus,
        runtime=settings.local_model_runtime,
        pinned=stage.pinned_settings(report.chosen),
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

    def pinned_settings(self, tuning: StageTuning) -> dict[str, int]:
        """Name the two settings rerank calibration searches.

        Args:
            tuning: The tuning to pin.

        Returns:
            ``RAG_RERANK_THREADS`` and ``RAG_RERANK_BATCH_SIZE``.

        Example:
            >>> stage.pinned_settings(StageTuning(threads=2, batch_size=2))
            {'RAG_RERANK_THREADS': 2, 'RAG_RERANK_BATCH_SIZE': 2}
        """
        return {
            RagSettings.variable_of("rerank_threads"): tuning.threads,
            RagSettings.variable_of("rerank_batch_size"): tuning.batch_size,
        }


class EmbedStage(BaseCalibratedStage[SentenceTransformerEmbedder]):
    """The embed stage on the in-process sentence-transformers model.

    Its sample is one question, embedded as a query the way the workflow embeds
    it, so the embedder puts the query prefix in front; a query is one text, so
    its batch size is never searched.

    Args:
        model: The model's files and runtime.
        pool: The shared inference pool.
        settings: ``embedding_query_prefix``, a setting of the embedder that
            each loaded adapter takes.

    Example:
        >>> stage = EmbedStage(model, pool, settings)
        >>> embedder = await stage.loaded(threads=2)
    """

    stage = "embed"

    def __init__(
        self, model: LocalModel, pool: Executor, settings: RagSettings
    ) -> None:
        self._model, self._pool = model, pool
        self._query_prefix = settings.embedding_query_prefix
        self._question = calibration_sample(1).question

    async def loaded(self, threads: int) -> SentenceTransformerEmbedder:
        """Load the embedding model for ``threads`` threads.

        Args:
            threads: Native threads one encode call may use.

        Returns:
            The embedder, at batch size 1 until ``rebatched``, with the query
            prefix of the settings.
        """
        return SentenceTransformerEmbedder.from_pretrained(
            self._model,
            self._pool,
            batch_size=1,
            threads=threads,
            query_prefix=self._query_prefix,
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
        """Embed the sample question as a query, as the workflow embeds a question.

        Args:
            adapter: The embedder under trial.
        """
        await adapter.embed_query(self._question)

    def pinned_settings(self, tuning: StageTuning) -> dict[str, int]:
        """Name the one setting embed calibration searches: its batch size is not.

        Args:
            tuning: The tuning to pin.

        Returns:
            ``RAG_EMBEDDING_THREADS``.

        Example:
            >>> stage.pinned_settings(StageTuning(threads=2, batch_size=64))
            {'RAG_EMBEDDING_THREADS': 2}
        """
        return {RagSettings.variable_of("embedding_threads"): tuning.threads}
