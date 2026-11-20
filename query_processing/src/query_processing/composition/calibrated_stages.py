"""The CPU-bound stages calibration times: rerank and embed on an in-process model.

Each stage says how its adapter loads for a thread count, takes another batch
size, serves one request of the usual size, on the path a real request takes,
and which settings of its component's ``local`` block pin a tuning;
``infrastructure/local_models/stage_calibration.py`` times them. They are
named calibrated stages so that no class shares a name with a stage of the
workflow. The ``local`` openers pick the candidates from a component's
``local`` block with ``candidates``: a setting above 0 is the one candidate, 0
searches.

Example:
    >>> calibrated = await calibrate(
    ...     CalibratedRerankStage(model, pool, passages=20),
    ...     threads=candidates(local.threads, budget.reranker_threads),
    ...     batch_sizes=candidates(local.batch_size, local.calibration_passages),
    ...     limits=local_models.calibration_limits,
    ... )
    >>> reranker = calibrated.adapter
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import TypeVar

from query_processing_core.retrieval.vector_store_record import ScoredVectorStoreRecord

from ..adapters.embedder.local.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from ..adapters.reranker.local.cross_encoder_reranker import CrossEncoderReranker
from ..config.workflow_settings import WorkflowSettings
from ..domain.runtime.calibration import (
    CalibrationReport,
    StageTuning,
    calibration_sample,
    halvings,
)
from ..infrastructure.local_models.local_model import LocalModel
from ..infrastructure.local_models.stage_calibration import (
    BaseCalibratedStage,
    calibrate,
)
from .local_model_resources import LocalModelResources

T = TypeVar("T")


@dataclass(frozen=True)
class ComponentCalibration:
    """One component's calibration on this machine, and the settings that pin it.

    Example:
        >>> calibration = await composition.calibration("reranker")
        >>> calibration.local
        {'threads': 2, 'batch_size': 2}
        >>> calibration.pinned
        {'QP_COMPONENTS__RERANKER__LOCAL__THREADS': 2,
         'QP_COMPONENTS__RERANKER__LOCAL__BATCH_SIZE': 2}
    """

    component: str  # the component's name in the workflow file
    report: CalibrationReport
    cpus: float  # the CPUs the thread budget counted here
    runtime: str  # what ran the model: openvino or torch
    local: dict[str, int]  # the settings of its local block that pin the choice

    @property
    def pinned(self) -> dict[str, int]:
        """The environment paths of the settings that pin the choice, and values."""
        return {
            WorkflowSettings.variable_of(
                "components", self.component, "local", name
            ): value
            for name, value in self.local.items()
        }


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


async def component_calibration(
    component: str,
    stage: BaseCalibratedStage[T],
    *,
    threads: Sequence[int],
    batch_sizes: Sequence[int],
    local_models: LocalModelResources,
    cpus: float,
) -> ComponentCalibration:
    """Calibrate ``stage`` over the candidates given, for ``qp-calibrate``.

    The adapter calibration returns is dropped here, so its model is released
    before the next component loads.

    Args:
        component: The component's name, whose environment paths pin the choice.
        stage: The stage to time.
        threads: Thread counts to try, the budget's share first.
        batch_sizes: Batch sizes to try, the whole request first.
        local_models: The calibration budget and rounds, and the runtime.
        cpus: The CPUs the budget counted, for the report.

    Returns:
        The report and the environment paths that pin its choice.

    Example:
        >>> await component_calibration("reranker", stage, threads=(2, 1),
        ...     batch_sizes=(20, 10), local_models=local_models, cpus=2.0)
    """
    calibrated = await calibrate(
        stage,
        threads=threads,
        batch_sizes=batch_sizes,
        limits=local_models.calibration_limits,
    )
    report = calibrated.report
    return ComponentCalibration(
        component,
        report,
        cpus=cpus,
        runtime=local_models.runtime,
        local=stage.pinned_settings(report.chosen),
    )


class CalibratedRerankStage(BaseCalibratedStage[CrossEncoderReranker]):
    """The rerank stage on the in-process cross-encoder, as calibration times it.

    Its sample is a question and ``passages`` passages, all kept: the model
    scores every candidate whatever it keeps, so the number kept does not
    change the time.

    Args:
        model: The model's files and runtime.
        pool: The shared inference pool.
        passages: How many passages the sample request reranks.

    Example:
        >>> stage = CalibratedRerankStage(model, pool, passages=20)
        >>> reranker = await stage.loaded(threads=2)
    """

    stage = "rerank"

    def __init__(self, model: LocalModel, pool: Executor, *, passages: int) -> None:
        self._model, self._pool = model, pool
        sample = calibration_sample(passages)
        self._question = sample.question
        self._passages = [
            ScoredVectorStoreRecord(
                text=text, metadata={"id": f"calibration-{i}"}, retrieval_score=0.0
            )
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
        passages = self._passages
        await adapter.rerank(self._question, passages, top_n=len(passages))

    def pinned_settings(self, tuning: StageTuning) -> dict[str, int]:
        """Name the two settings rerank calibration searches.

        Args:
            tuning: The tuning to pin.

        Returns:
            The ``local`` block's ``threads`` and ``batch_size``.

        Example:
            >>> stage.pinned_settings(StageTuning(threads=2, batch_size=2))
            {'threads': 2, 'batch_size': 2}
        """
        return {"threads": tuning.threads, "batch_size": tuning.batch_size}


class CalibratedEmbedStage(BaseCalibratedStage[SentenceTransformerEmbedder]):
    """The embed stage on the in-process sentence-transformers model, as timed.

    Its sample is one question, embedded as a query the way the workflow embeds
    it, so the embedder puts the query prefix in front; a query is one text, so
    its batch size is never searched.

    Args:
        model: The model's files and runtime.
        pool: The shared inference pool.
        query_prefix: The embedder's query prefix, which each loaded adapter
            takes.
        normalize_embeddings: Whether each loaded adapter scales its vectors
            to unit length: calibration times the encode the App will run.

    Example:
        >>> stage = CalibratedEmbedStage(
        ...     model, pool, query_prefix="query: ", normalize_embeddings=False)
        >>> embedder = await stage.loaded(threads=2)
    """

    stage = "embed"

    def __init__(
        self,
        model: LocalModel,
        pool: Executor,
        *,
        query_prefix: str,
        normalize_embeddings: bool,
    ) -> None:
        self._model, self._pool = model, pool
        self._query_prefix = query_prefix
        # Calibration times the encode the App will run, normalised or not.
        self._normalize = normalize_embeddings
        self._question = calibration_sample(1).question

    async def loaded(self, threads: int) -> SentenceTransformerEmbedder:
        """Load the embedding model for ``threads`` threads.

        Args:
            threads: Native threads one encode call may use.

        Returns:
            The embedder, at batch size 1 until ``rebatched``, with the query
            prefix.
        """
        return SentenceTransformerEmbedder.from_pretrained(
            self._model,
            self._pool,
            batch_size=1,
            threads=threads,
            query_prefix=self._query_prefix,
            normalize_embeddings=self._normalize,
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
            The ``local`` block's ``threads``.

        Example:
            >>> stage.pinned_settings(StageTuning(threads=2, batch_size=64))
            {'threads': 2}
        """
        return {"threads": tuning.threads}
