"""The embedder on the ``local`` backend: a sentence-transformers model in-process.

Its thread count, left at 0, is calibrated when it opens;
``calibrate_embedder_on_local`` searches it on demand, for ``rag-calibrate``.

Example:
    >>> embedder = await open_embedder_on_local(embedder_settings, shared)
"""

from __future__ import annotations

from ...adapters.embedder.base_embedder import BaseEmbedder
from ...config.component_settings import EmbedderSettings
from ...domain.calibration import halvings
from ...infrastructure.local_models.openvino_export import export_embedder
from ...infrastructure.local_models.stage_calibration import calibrate
from ..calibrated_stages import (
    CalibratedEmbedStage,
    ComponentCalibration,
    candidates,
    component_calibration,
)
from ..local_model_resources import LocalModelResources
from .component_openers import EMBEDDER_OPENERS, SharedResources


@EMBEDDER_OPENERS.register("local")
async def open_embedder_on_local(
    component: EmbedderSettings, shared: SharedResources
) -> BaseEmbedder:
    """Load the in-process embedder, calibrating its threads when left at 0.

    Args:
        component: The embedder's source id, query prefix and ``local`` block.
        shared: The thread budget, the model pool and the calibration limits.

    Returns:
        The sentence-transformers embedder, at the fastest thread count tried.

    Example:
        >>> embedder = await open_embedder_on_local(embedder_settings, shared)
    """
    local, local_models = component.local, shared.local_models
    budget = await local_models.thread_budget()
    # A query is one text, so only the threads are searched.
    calibrated = await calibrate(
        await _calibrated_stage(component, local_models),
        threads=candidates(local.threads, budget.embedder_threads),
        batch_sizes=(local.batch_size,),
        limits=local_models.calibration_limits,
    )
    return calibrated.adapter


async def calibrate_embedder_on_local(
    name: str, component: EmbedderSettings, local_models: LocalModelResources
) -> ComponentCalibration:
    """Calibrate the in-process embedder here, searching its threads.

    A pinned thread count is searched too: revisiting it is the point. The batch
    size stays as set: a query is one text. The model is dropped once timed.

    Args:
        name: The component's name, whose environment paths pin the choice.
        component: The embedder's source id, query prefix and ``local`` block.
        local_models: The machine's thread budget, the model pool, the
            calibration limits and the runtime.

    Returns:
        The report, the CPUs and runtime it ran on, and the environment paths
        that pin the choice.

    Example:
        >>> (await calibrate_embedder_on_local("embedder", embedder, local_models)
        ... ).pinned
        {'RAG_COMPONENTS__EMBEDDER__LOCAL__THREADS': 2}
    """
    machine = await local_models.unpinned_budget()
    return await component_calibration(
        name,
        await _calibrated_stage(component, local_models),
        threads=halvings(machine.budget.embedder_threads),
        batch_sizes=(component.local.batch_size,),
        local_models=local_models,
        cpus=machine.cpus,
    )


async def _calibrated_stage(
    component: EmbedderSettings, local_models: LocalModelResources
) -> CalibratedEmbedStage:
    pool = await local_models.model_pool()
    model = await local_models.local_model(component.source_id, export_embedder)
    return CalibratedEmbedStage(model, pool, query_prefix=component.query_prefix)
