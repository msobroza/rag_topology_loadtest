"""What the in-process models share: the thread budget, the pool and their files.

The thread budget comes from the first CPU counter that answers
(``QP_LOCAL_MODELS__CPUS``, the Domino tier, the container; see
``infrastructure/local_models/cpu_count.py``), divided by Uvicorn's
workers (``QP_UVICORN_WORKERS``): each is a process loading its own models.
The pool is created on first need and shut down with the scope. Under the
``openvino`` runtime a model whose folder is missing is exported there first,
at the settings' weight format, on the pool, since it takes minutes
(``infrastructure/local_models/model_provisioning.py``).

Example:
    >>> local_models = LocalModelResources(settings, scope, http, ContainerCpus())
    >>> model = await local_models.local_model("org/name", export_embedder)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from ..config.apps.uvicorn_settings import UvicornSettings
from ..config.workflow_settings import WorkflowSettings
from ..domain.runtime.thread_budget import ThreadBudget, plan_thread_budget
from ..infrastructure.calibration.timed_trials import CalibrationLimits
from ..infrastructure.local_models.cpu_count import (
    BaseCpuCounter,
    DominoHardwareTierCpus,
    FixedCpus,
    first_cpu_count,
    work_out_thread_budget,
)
from ..infrastructure.local_models.local_model import LocalModel, OpenVinoExports
from ..infrastructure.local_models.model_pool import model_executor
from ..infrastructure.local_models.model_provisioning import (
    Exporter,
    PackageCheck,
    provision_local_model,
)
from ..infrastructure.local_models.native_threads import settle_torch_thread
from ..infrastructure.local_models.openvino_export import require_openvino
from .http_resources import HttpResources
from .resource_scope import OWNED, ResourceScope


class UnpinnedBudget(NamedTuple):
    """The CPUs counted here, and their budget as if no thread count were pinned."""

    cpus: float
    budget: ThreadBudget


class _ShareOfOneWorker(BaseCpuCounter):
    """A counter's answer divided among the processes serving the app.

    Args:
        counter: The CPU source whose answer is divided; its name is kept.
        workers: Uvicorn's processes, each loading its own models.
    """

    def __init__(self, counter: BaseCpuCounter, workers: int) -> None:
        self.name, self._counter, self._workers = counter.name, counter, workers

    async def cpus(self) -> float | None:
        """Return the counter's CPUs divided by the workers.

        Returns:
            One worker's share, or ``None`` when the counter cannot say.
        """
        cpus = await self._counter.cpus()
        return None if cpus is None else cpus / self._workers


class LocalModelResources:
    """The thread budget, the model pool and the files of the in-process models.

    ``runtime`` and ``calibration_limits`` are read from the settings once: what
    runs the models, and the budget and rounds every calibration runs under.

    Args:
        settings: The CPUs and workers, the runtime, its folder and its
            weight format (its ``local_models`` block), the calibration budget
            and rounds (its ``calibration`` block).
        scope: Opens the budget and the pool once and shuts the pool down.
        http: The shared HTTP client, for the Domino hardware tier lookup.
        container_cpus: Counts the container's CPUs when nothing earlier in the
            chain answers.
        require_openvino_packages: Raises when the ``openvino`` runtime's
            packages are missing; checked before each model is provisioned.

    Example:
        >>> budget = await local_models.thread_budget()  # openvino, 8 CPUs
        >>> budget.workers
        1
    """

    def __init__(
        self,
        settings: WorkflowSettings,
        scope: ResourceScope,
        http: HttpResources,
        container_cpus: BaseCpuCounter,
        require_openvino_packages: PackageCheck = require_openvino,
    ) -> None:
        self._settings, self._scope, self._http = settings, scope, http
        self._container_cpus = container_cpus
        self.runtime = settings.local_models.runtime  # openvino or torch
        # The CPUs a core holds: the cores are the CPUs counted divided by it.
        self.threads_per_core = settings.local_models.threads_per_core
        # Every CPU-bound stage is calibrated under the same budget and rounds.
        calibration = settings.calibration
        self.calibration_limits = CalibrationLimits(
            budget_s=calibration.budget_s, rounds=calibration.rounds
        )
        self._require_openvino_packages = require_openvino_packages
        # Uvicorn's processes, each loading its own models on its share of the CPUs.
        self._server_workers = UvicornSettings().workers

    async def thread_budget(self) -> ThreadBudget:
        """Return how the virtual CPUs are split, worked out on the first request.

        A component's ``local`` block may pin its own thread count, which then
        replaces its share: its opener applies it.

        Returns:
            The workers and each model's share of threads.

        Example:
            >>> budget = await local_models.thread_budget()  # openvino, 8 CPUs
            >>> budget.embedder_threads
            8
        """
        return await self._scope.once("thread_budget", OWNED, self._work_out_budget)

    async def unpinned_budget(self) -> UnpinnedBudget:
        """Return the CPUs and the budget's shares, whatever the thread settings say.

        The tops of a calibration that revisits a pinned thread count are the
        shares themselves.

        Returns:
            The CPUs counted and the budget planned from them alone.

        Example:
            >>> (await local_models.unpinned_budget()).budget.reranker_threads
            8
        """
        return await self._scope.once(
            "unpinned_budget", OWNED, self._work_out_unpinned_budget
        )

    async def model_pool(self) -> ThreadPoolExecutor:
        """Return the pool the models run on, started on the first request.

        Returns:
            The pool, shut down when the scope ends.

        Example:
            >>> pool = await local_models.model_pool()
        """
        return await self._scope.once("model_pool", OWNED, self._start_model_pool)

    async def local_model(self, model_id: str, export: Exporter) -> LocalModel:
        """Return what a model loads from, exporting it first when it must be.

        Args:
            model_id: The model's Hugging Face id or local path.
            export: Converts this kind of model to OpenVINO files, at the weight
                format the settings name.

        Returns:
            The files the model loads from (the export of the settings' weight
            format, under ``openvino``) and what runs it.

        Example:
            >>> await local_models.local_model("org/rerank", export_reranker)
        """
        local_models = self._settings.local_models
        return await provision_local_model(
            model_id,
            runtime=local_models.runtime,
            exports=OpenVinoExports(
                local_models.openvino_dir,
                local_models.openvino_weight_format,
                local_models.openvino.chosen(),
            ),
            export=export,
            require_packages=self._require_openvino_packages,
            executor=await self.model_pool(),
        )

    async def _start_model_pool(self) -> ThreadPoolExecutor:
        workers = (await self.thread_budget()).workers
        # Only real local models reach here; see settle_torch_thread for why.
        pool = model_executor(workers, on_thread_start=settle_torch_thread)
        self._scope.exit_stack("model_pool").callback(pool.shutdown, wait=False)
        return pool

    async def _work_out_unpinned_budget(self) -> UnpinnedBudget:
        local_models = self._settings.local_models
        cpus, _source = await first_cpu_count(await self._cpu_counters())
        budget = plan_thread_budget(
            cpus, max_workers=local_models.workers, runtime=local_models.runtime
        )
        return UnpinnedBudget(cpus, budget)

    async def _work_out_budget(self) -> ThreadBudget:
        local_models = self._settings.local_models
        return await work_out_thread_budget(
            await self._cpu_counters(),
            max_workers=local_models.workers,
            runtime=local_models.runtime,
        )

    async def _cpu_counters(self) -> list[BaseCpuCounter]:
        settings = self._settings
        counters: list[BaseCpuCounter] = [FixedCpus(settings.local_models.cpus)]
        if settings.domino_api_proxy and settings.domino_hardware_tier_id:
            counters.append(
                DominoHardwareTierCpus(
                    await self._http.client(),
                    settings.domino_api_proxy,
                    settings.domino_hardware_tier_id,
                    self._http.policy,
                )
            )
        counters.append(self._container_cpus)
        return [_ShareOfOneWorker(c, self._server_workers) for c in counters]
