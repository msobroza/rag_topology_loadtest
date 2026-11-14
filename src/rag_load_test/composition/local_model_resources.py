"""What the in-process models share: the thread budget, the pool and their files.

The thread budget comes from the first CPU counter that answers
(``RAG_MODEL_CPUS``, the Domino tier, the container; see
``infrastructure/local_models/cpu_count.py``). The pool is created on first need
and shut down with the scope. Under the ``openvino`` runtime a model whose
folder is missing is exported there first, on the pool, since it takes minutes
(``infrastructure/local_models/model_provisioning.py``).

Example:
    >>> local_models = LocalModelResources(settings, scope, http, ContainerCpus())
    >>> model = await local_models.local_model("org/name", export_embedder)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..config.rag_settings import RagSettings
from ..domain.thread_budget import ThreadBudget
from ..infrastructure.local_models.cpu_count import (
    BaseCpuCounter,
    DominoHardwareTierCpus,
    FixedCpus,
    work_out_thread_budget,
)
from ..infrastructure.local_models.local_model import LocalModel
from ..infrastructure.local_models.model_pool import model_executor
from ..infrastructure.local_models.model_provisioning import (
    Exporter,
    provision_local_model,
)
from ..infrastructure.local_models.native_threads import settle_torch_thread
from .http_resources import HttpResources
from .resource_scope import OWNED, ResourceScope


class LocalModelResources:
    """The thread budget, the model pool and the files of the in-process models.

    Args:
        settings: The CPU, worker and thread counts, the runtime and its folder.
        scope: Opens the budget and the pool once and shuts the pool down.
        http: The shared HTTP client, for the Domino hardware tier lookup.
        container_cpus: Counts the container's CPUs when nothing earlier in the
            chain answers.

    Example:
        >>> budget = await local_models.thread_budget()
        >>> budget.workers
        4
    """

    def __init__(
        self,
        settings: RagSettings,
        scope: ResourceScope,
        http: HttpResources,
        container_cpus: BaseCpuCounter,
    ) -> None:
        self._settings, self._scope, self._http = settings, scope, http
        self._container_cpus = container_cpus

    async def thread_budget(self) -> ThreadBudget:
        """Return how the virtual CPUs are split, worked out on the first request.

        Returns:
            The workers and the threads of each model.

        Example:
            >>> (await local_models.thread_budget()).embedder_threads
            2
        """
        return await self._scope.once("thread_budget", OWNED, self._work_out_budget)

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
            export: Converts this kind of model to OpenVINO files.

        Returns:
            The files the model loads from and what runs it.

        Example:
            >>> await local_models.local_model("org/rerank", export_reranker)
        """
        settings = self._settings
        return await provision_local_model(
            model_id,
            runtime=settings.local_model_runtime,
            models_dir=settings.openvino_models_dir,
            export=export,
            executor=await self.model_pool(),
        )

    async def _start_model_pool(self) -> ThreadPoolExecutor:
        workers = (await self.thread_budget()).workers
        # Only real local models reach here; see settle_torch_thread for why.
        pool = model_executor(workers, on_thread_start=settle_torch_thread)
        self._scope.exit_stack("model_pool").callback(pool.shutdown, wait=False)
        return pool

    async def _work_out_budget(self) -> ThreadBudget:
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
        return await work_out_thread_budget(
            [*counters, self._container_cpus],
            max_workers=settings.model_workers,
            embedder_threads=settings.embedding_threads,
            reranker_threads=settings.rerank_threads,
        )
