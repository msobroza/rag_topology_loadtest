"""What the in-process models share: the thread budget, the pool and their files.

The thread budget comes from the first CPU counter that answers
(``RAG_MODEL_CPUS``, the Domino tier, the container; see
``infrastructure/local_models/cpu_count.py``). The pool is created on first need
and shut down with the scope. Under the ``openvino`` runtime a model whose
folder is missing is exported there first, at the settings' weight format, on
the pool, since it takes minutes
(``infrastructure/local_models/model_provisioning.py``).

Example:
    >>> local_models = LocalModelResources(settings, scope, http, ContainerCpus())
    >>> model = await local_models.local_model("org/name", export_embedder)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from ..config.rag_settings import RagSettings
from ..domain.thread_budget import ThreadBudget, plan_thread_budget
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


class LocalModelResources:
    """The thread budget, the model pool and the files of the in-process models.

    Args:
        settings: The CPU, worker and thread counts, the runtime and its folder.
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
        settings: RagSettings,
        scope: ResourceScope,
        http: HttpResources,
        container_cpus: BaseCpuCounter,
        require_openvino_packages: PackageCheck = require_openvino,
    ) -> None:
        self._settings, self._scope, self._http = settings, scope, http
        self._container_cpus = container_cpus
        self._require_openvino_packages = require_openvino_packages

    async def thread_budget(self) -> ThreadBudget:
        """Return how the virtual CPUs are split, worked out on the first request.

        Returns:
            The workers and the threads of each model.

        Example:
            >>> budget = await local_models.thread_budget()  # openvino, 8 CPUs
            >>> budget.embedder_threads
            8
        """
        return await self._scope.once("thread_budget", OWNED, self._work_out_budget)

    async def unpinned_budget(self) -> UnpinnedBudget:
        """Return the CPUs and the budget's shares, whatever the thread settings say.

        A pinned ``RAG_*_THREADS`` replaces its share in ``thread_budget``; the
        tops of a calibration that revisits it are the shares themselves.

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
        settings = self._settings
        return await provision_local_model(
            model_id,
            runtime=settings.local_model_runtime,
            exports=OpenVinoExports(
                settings.openvino_models_dir, settings.openvino_weight_format
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
        settings = self._settings
        cpus, _source = await first_cpu_count(await self._cpu_counters())
        budget = plan_thread_budget(
            cpus,
            max_workers=settings.model_workers,
            runtime=settings.local_model_runtime,
        )
        return UnpinnedBudget(cpus, budget)

    async def _work_out_budget(self) -> ThreadBudget:
        settings = self._settings
        return await work_out_thread_budget(
            await self._cpu_counters(),
            max_workers=settings.model_workers,
            runtime=settings.local_model_runtime,
            embedder_threads=settings.embedding_threads,
            reranker_threads=settings.rerank_threads,
        )

    async def _cpu_counters(self) -> list[BaseCpuCounter]:
        settings = self._settings
        counters: list[BaseCpuCounter] = [FixedCpus(settings.model_cpus)]
        if settings.domino_api_proxy and settings.domino_hardware_tier_id:
            counters.append(
                DominoHardwareTierCpus(
                    await self._http.client(),
                    settings.domino_api_proxy,
                    settings.domino_hardware_tier_id,
                    self._http.policy,
                )
            )
        return [*counters, self._container_cpus]
