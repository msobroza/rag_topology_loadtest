"""Builds the embedder from the settings: ``RAG_EMBEDDING_BACKEND`` picks which.

Example:
    >>> embedder = await EmbedderFactory(settings, scope, http, local_models).open()
"""

from __future__ import annotations

from collections.abc import Mapping

from ..adapters.embedder.api.api_embedder import open_api_embedder
from ..adapters.embedder.base_embedder import BaseEmbedder
from ..adapters.embedder.fake.fake_embedder import FakeEmbedder
from ..config.rag_settings import RagSettings
from ..domain.calibration import halvings
from ..infrastructure.local_models.openvino_export import export_embedder
from ..infrastructure.local_models.stage_calibration import calibrate
from .base_port_factory import BasePortFactory
from .calibrated_stages import (
    ComponentCalibration,
    EmbedStage,
    calibration_limits,
    candidates,
    component_calibration,
)
from .http_resources import HttpResources
from .local_model_resources import LocalModelResources
from .resource_scope import Opener, ResourceScope


class EmbedderFactory(BasePortFactory[BaseEmbedder]):
    """The embedder of each Backend: ``local``, ``api`` or ``fake``.

    Args:
        settings: The embedding Backend, model, endpoint, query prefix, batch
            size and threads (0 = calibrated at start-up).
        scope: Opens the embedder once.
        http: The shared HTTP client, for the ``api`` Backend.
        local_models: The thread budget, pool and files, for the ``local`` one.
        supplied: A ready-made embedder, used as given; None to build one.

    Example:
        >>> await EmbedderFactory(settings, scope, http, local_models).open()
    """

    port = "embedder"

    def __init__(
        self,
        settings: RagSettings,
        scope: ResourceScope,
        http: HttpResources,
        local_models: LocalModelResources,
        supplied: BaseEmbedder | None = None,
    ) -> None:
        super().__init__(settings, scope, supplied)
        self._http, self._local_models = http, local_models

    def backend(self) -> str:
        """Return ``RAG_EMBEDDING_BACKEND``, as the topology resolved it.

        Returns:
            ``local``, ``api`` or ``fake``.
        """
        return self._settings.embedding_backend

    def openers(self) -> Mapping[str, Opener[BaseEmbedder]]:
        """Return what opens the embedder of each Backend.

        Returns:
            One entry per embedding Backend value.
        """
        return {"local": self._local, "api": self._api, "fake": self._fake}

    async def open_local(self) -> BaseEmbedder:
        """Open the in-process embedder whatever the Backend says.

        A model server is what the ``api`` Backend calls, so it always loads the
        local model.

        Returns:
            The supplied embedder if one was given, else the sentence-transformers
            one.

        Example:
            >>> embedder = await embedder_factory.open_local()
        """
        if self._supplied is not None:
            return self._supplied
        return await self._local()

    async def calibration(self) -> ComponentCalibration:
        """Calibrate the in-process embedder here, searching its threads.

        A pinned ``RAG_EMBEDDING_THREADS`` is searched too: revisiting it is the
        point. The batch size stays as set: a query is one text. The model is
        dropped once timed.

        Returns:
            The report, the CPUs and runtime it ran on, and the pinning settings.

        Example:
            >>> (await embedder_factory.calibration()).pinned
            {'RAG_EMBEDDING_THREADS': 2}
        """
        settings = self._settings
        machine = await self._local_models.unpinned_budget()
        return await component_calibration(
            await self._stage(),
            threads=halvings(machine.budget.embedder_threads),
            batch_sizes=(settings.embedding_batch_size,),
            settings=settings,
            cpus=machine.cpus,
        )

    async def _local(self) -> BaseEmbedder:
        return await self._scope.once("local_embedder", "local", self._load)

    async def _stage(self) -> EmbedStage:
        settings, local_models = self._settings, self._local_models
        pool = await local_models.model_pool()
        model = await local_models.local_model(
            settings.embedding_model, export_embedder
        )
        return EmbedStage(model, pool, settings)

    async def _load(self) -> BaseEmbedder:
        settings = self._settings
        budget = await self._local_models.thread_budget()
        # A query is one text, so only the threads are searched.
        calibrated = await calibrate(
            await self._stage(),
            threads=candidates(settings.embedding_threads, budget.embedder_threads),
            batch_sizes=(settings.embedding_batch_size,),
            limits=calibration_limits(settings),
        )
        return calibrated.adapter

    async def _api(self) -> BaseEmbedder:
        return open_api_embedder(
            await self._http.client(),
            self._settings.embedding_api(),
            tokens=self._http.run_tokens,
            policy=self._http.policy,
            batch_size=self._settings.embedding_batch_size,
            query_prefix=self._settings.embedding_query_prefix,
        )

    async def _fake(self) -> BaseEmbedder:
        return FakeEmbedder()
