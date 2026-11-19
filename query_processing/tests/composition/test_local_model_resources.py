"""The thread budget of one process: the CPUs counted, shared by Uvicorn's workers.

Each worker of ``qp-serve`` or ``qp-model-server`` is a process of its own that
loads its own models, so each plans its threads on its share of the CPUs. The
count comes from those two commands alone: any other runs as one process.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from cpu_fakes import FakeCpuCounter
from query_processing.composition.http_resources import HttpResources
from query_processing.composition.local_model_resources import LocalModelResources
from query_processing.composition.resource_scope import ResourceScope
from query_processing.config.workflow_settings import WorkflowSettings


def resources_on(
    settings_factory: Callable[..., WorkflowSettings],
    scope: ResourceScope,
    cpus: float,
    **processes: int,
) -> LocalModelResources:
    """The in-process models' resources on ``cpus`` CPUs, under ``openvino``."""
    settings = settings_factory(local_models__runtime="openvino")
    http = HttpResources(settings, scope)
    container = FakeCpuCounter(cpus, "container")
    return LocalModelResources(settings, scope, http, container, **processes)


@pytest.mark.parametrize(("workers", "threads"), [(1, 8), (2, 4), (3, 2)])
async def test_the_workers_share_the_cpus(
    settings_factory: Callable[..., WorkflowSettings], workers: int, threads: int
) -> None:
    scope = ResourceScope("test")
    local_models = resources_on(settings_factory, scope, 8, server_processes=workers)

    async with scope:
        budget = await local_models.thread_budget()
        unpinned = await local_models.unpinned_budget()

    assert (budget.workers, budget.reranker_threads) == (1, threads)
    assert budget.embedder_threads == threads
    assert unpinned.budget.reranker_threads == threads
    assert unpinned.cpus == 8 / workers


async def test_one_process_keeps_every_cpu_whatever_the_uvicorn_workers_variable(
    settings_factory: Callable[..., WorkflowSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    # qp-ingest and qp-calibrate run as one process, whatever .env gives Uvicorn.
    monkeypatch.setenv("QP_UVICORN_WORKERS", "2")
    scope = ResourceScope("test")
    local_models = resources_on(settings_factory, scope, 8)

    async with scope:
        unpinned = await local_models.unpinned_budget()

    assert (unpinned.cpus, unpinned.budget.reranker_threads) == (8, 8)
