"""The reranker on the ``local`` backend: a cross-encoder model in this process.

Its batch size and thread count, left at 0, are calibrated when it opens;
``calibrate_reranker_on_local`` searches both on demand, for ``rag-calibrate``.

Example:
    >>> reranker = await open_reranker_on_local(components.reranker, shared)
"""

from __future__ import annotations

from ...adapters.reranker.base_reranker import BaseReranker
from ...config.component_settings import RerankerSettings
from ...domain.calibration import halvings
from ...infrastructure.local_models.openvino_export import export_reranker
from ...infrastructure.local_models.stage_calibration import calibrate
from ..calibrated_stages import (
    CalibratedRerankStage,
    ComponentCalibration,
    candidates,
    component_calibration,
)
from ..local_model_resources import LocalModelResources
from .component_openers import RERANKER_OPENERS, SharedResources


@RERANKER_OPENERS.register("local")
async def open_reranker_on_local(
    component: RerankerSettings, shared: SharedResources
) -> BaseReranker:
    """Load the in-process reranker, calibrating what its ``local`` block leaves at 0.

    Args:
        component: The reranker's source id and ``local`` block.
        shared: The thread budget, the model pool and the calibration limits.

    Returns:
        The cross-encoder reranker, at the fastest batch size and threads tried.

    Example:
        >>> reranker = await open_reranker_on_local(components.reranker, shared)
    """
    local, local_models = component.local, shared.local_models
    budget = await local_models.thread_budget()
    # A batch is padded to its longest passage, so the request's size bounds
    # the batch sizes worth trying.
    calibrated = await calibrate(
        await _calibrated_stage(component, local_models),
        threads=candidates(local.threads, budget.reranker_threads),
        batch_sizes=candidates(local.batch_size, local.calibration_passages),
        limits=local_models.calibration_limits,
    )
    return calibrated.adapter


async def calibrate_reranker_on_local(
    component: RerankerSettings, local_models: LocalModelResources
) -> ComponentCalibration:
    """Calibrate the in-process reranker here, searching both its parameters.

    A pinned thread count or batch size is searched too: revisiting it is the
    point. The model is dropped once timed.

    Args:
        component: The reranker's source id and ``local`` block.
        local_models: The machine's thread budget, the model pool, the
            calibration limits and the runtime.

    Returns:
        The report, the CPUs and runtime it ran on, and the pinning settings.

    Example:
        >>> (await calibrate_reranker_on_local(reranker, local_models)).pinned
        {'RAG_RERANK_THREADS': 2, 'RAG_RERANK_BATCH_SIZE': 2}
    """
    machine = await local_models.unpinned_budget()
    return await component_calibration(
        await _calibrated_stage(component, local_models),
        threads=halvings(machine.budget.reranker_threads),
        batch_sizes=halvings(component.local.calibration_passages),
        local_models=local_models,
        cpus=machine.cpus,
    )


async def _calibrated_stage(
    component: RerankerSettings, local_models: LocalModelResources
) -> CalibratedRerankStage:
    local = component.local
    pool = await local_models.model_pool()
    model = await local_models.local_model(component.source_id, export_reranker)
    return CalibratedRerankStage(
        model, pool, passages=local.calibration_passages, top_n=local.calibration_top_n
    )
