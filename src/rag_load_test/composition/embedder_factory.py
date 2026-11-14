"""Builds the embedder from the settings: ``RAG_EMBEDDING_BACKEND`` picks which.

Example:
    >>> embedder = await EmbedderFactory(settings, scope, http, local_models).open()
"""

from __future__ import annotations

from collections.abc import Mapping

from ..adapters.embedder.api.api_embedder import open_api_embedder
from ..adapters.embedder.base_embedder import BaseEmbedder
from ..adapters.embedder.fake.fake_embedder import FakeEmbedder
from ..adapters.embedder.local.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from ..config.rag_settings import RagSettings
from ..infrastructure.local_models.openvino_export import export_embedder
from .base_port_factory import BasePortFactory
from .http_resources import HttpResources
from .local_model_resources import LocalModelResources
from .resource_scope import Opener, ResourceScope


class EmbedderFactory(BasePortFactory[BaseEmbedder]):
    """The embedder of each Backend: ``local``, ``api`` or ``fake``.

    Args:
        settings: The embedding Backend, model, endpoint and batch size.
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

    async def _local(self) -> BaseEmbedder:
        return await self._scope.once("local_embedder", "local", self._load)

    async def _load(self) -> BaseEmbedder:
        settings, local_models = self._settings, self._local_models
        budget, pool = (
            await local_models.thread_budget(),
            await local_models.model_pool(),
        )
        model = await local_models.local_model(
            settings.embedding_model, export_embedder
        )
        return SentenceTransformerEmbedder.from_pretrained(
            model,
            pool,
            batch_size=settings.embedding_batch_size,
            threads=budget.embedder_threads,
        )

    async def _api(self) -> BaseEmbedder:
        return open_api_embedder(
            await self._http.client(),
            self._settings.embedding_api(),
            tokens=self._http.run_tokens,
            policy=self._http.policy,
            batch_size=self._settings.embedding_batch_size,
        )

    async def _fake(self) -> BaseEmbedder:
        return FakeEmbedder()
