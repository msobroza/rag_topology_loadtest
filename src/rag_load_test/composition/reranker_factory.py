"""Builds the reranker from the settings: ``RAG_RERANK_BACKEND`` picks which.

Example:
    >>> reranker = await RerankerFactory(settings, scope, http, local_models).open()
"""

from __future__ import annotations

from collections.abc import Mapping

from ..adapters.reranker.api.api_reranker import open_api_reranker
from ..adapters.reranker.base_reranker import BaseReranker
from ..adapters.reranker.fake.fake_reranker import FakeReranker
from ..adapters.reranker.none.no_reranker import NoReranker
from ..config.rag_settings import RagSettings
from ..domain.calibration import halvings
from ..infrastructure.local_models.openvino_export import export_reranker
from ..infrastructure.local_models.stage_calibration import calibrate
from .base_port_factory import BasePortFactory
from .calibrated_stages import (
    ComponentCalibration,
    RerankStage,
    calibration_limits,
    candidates,
    component_calibration,
)
from .http_resources import HttpResources
from .local_model_resources import LocalModelResources
from .resource_scope import Opener, ResourceScope


class RerankerFactory(BasePortFactory[BaseReranker]):
    """The reranker of each Backend: ``local``, ``api``, ``fake`` or ``none``.

    Args:
        settings: The rerank Backend, model, endpoint, batch size and threads
            (0 = calibrated at start-up).
        scope: Opens the reranker once.
        http: The shared HTTP client, for the ``api`` Backend.
        local_models: The thread budget, pool and files, for the ``local`` one.
        supplied: A ready-made reranker, used as given; None to build one.

    Example:
        >>> await RerankerFactory(settings, scope, http, local_models).open()
    """

    port = "reranker"

    def __init__(
        self,
        settings: RagSettings,
        scope: ResourceScope,
        http: HttpResources,
        local_models: LocalModelResources,
        supplied: BaseReranker | None = None,
    ) -> None:
        super().__init__(settings, scope, supplied)
        self._http, self._local_models = http, local_models

    def backend(self) -> str:
        """Return ``RAG_RERANK_BACKEND``, as the topology resolved it.

        Returns:
            ``local``, ``api``, ``fake`` or ``none``.
        """
        return self._settings.rerank_backend

    def openers(self) -> Mapping[str, Opener[BaseReranker]]:
        """Return what opens the reranker of each Backend.

        Returns:
            One entry per rerank Backend value.
        """
        return {
            "local": self._local,
            "api": self._api,
            "fake": self._fake,
            "none": self._none,
        }

    async def open_local(self) -> BaseReranker:
        """Open the in-process reranker whatever the Backend says.

        A model server is what the ``api`` Backend calls, so it always loads the
        local model.

        Returns:
            The supplied reranker if one was given, else the cross-encoder one.

        Example:
            >>> reranker = await reranker_factory.open_local()
        """
        if self._supplied is not None:
            return self._supplied
        return await self._local()

    async def calibration(self) -> ComponentCalibration:
        """Calibrate the in-process reranker here, searching both its parameters.

        A pinned ``RAG_RERANK_THREADS`` or ``RAG_RERANK_BATCH_SIZE`` is searched
        too: revisiting it is the point. The model is dropped once timed.

        Returns:
            The report, the CPUs and runtime it ran on, and the pinning settings.

        Example:
            >>> (await reranker_factory.calibration()).report.chosen
            StageTuning(threads=2, batch_size=2)
        """
        settings = self._settings
        machine = await self._local_models.unpinned_budget()
        return await component_calibration(
            await self._stage(),
            threads=halvings(machine.budget.reranker_threads),
            batch_sizes=halvings(settings.top_k_retrieve),
            settings=settings,
            cpus=machine.cpus,
        )

    async def _local(self) -> BaseReranker:
        return await self._scope.once("local_reranker", "local", self._load)

    async def _stage(self) -> RerankStage:
        settings, local_models = self._settings, self._local_models
        pool = await local_models.model_pool()
        model = await local_models.local_model(settings.rerank_model, export_reranker)
        return RerankStage(model, pool, settings)

    async def _load(self) -> BaseReranker:
        settings = self._settings
        budget = await self._local_models.thread_budget()
        # A batch is padded to its longest passage, so the request's size bounds
        # the batch sizes worth trying.
        calibrated = await calibrate(
            await self._stage(),
            threads=candidates(settings.rerank_threads, budget.reranker_threads),
            batch_sizes=candidates(settings.rerank_batch_size, settings.top_k_retrieve),
            limits=calibration_limits(settings),
        )
        return calibrated.adapter

    async def _api(self) -> BaseReranker:
        return open_api_reranker(
            await self._http.client(),
            self._settings.rerank_api(),
            tokens=self._http.run_tokens,
            policy=self._http.policy,
        )

    async def _fake(self) -> BaseReranker:
        return FakeReranker()

    async def _none(self) -> BaseReranker:
        return NoReranker()
