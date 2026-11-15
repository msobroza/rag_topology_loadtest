"""The stages OVMS serves, as calibration times them: rerank and embed.

Each candidate tuning gets a serving folder written for it and one ``ovms``
start (``infrastructure/ovms/``); the sample's requests are then sent at once,
``RAG_CALIBRATION_USERS`` of them, through the same ``api`` adapters the
workflow app calls OVMS with, on OVMS's /v3 routes and with the alias as model.
``infrastructure/calibration/served_stage_calibration.py`` times them.

Example:
    >>> stage = open_ovms_stage("reranker", settings, client, models_dir, folder)
    >>> report = await calibrate_served(stage, streams=(1, 2), threads=(2, 1), ...)
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import cached_property
from pathlib import Path
from typing import Generic, Protocol, TypeVar

import httpx

from ..adapters.embedder.api.api_embedder import ApiEmbedder, open_api_embedder
from ..adapters.reranker.api.api_reranker import ApiReranker, open_api_reranker
from ..config.rag_settings import RagSettings
from ..domain.api_endpoint import (
    MODEL_SERVER_EMBEDDINGS_PATH,
    MODEL_SERVER_RERANK_PATH,
    ApiEndpoint,
)
from ..domain.calibration import StageTuning, calibration_sample
from ..domain.passages import ScoredPassage
from ..domain.roles import ModelServerPart
from ..infrastructure.calibration.served_stage_calibration import BaseServedStage
from ..infrastructure.endpoint.open_endpoint_client import RunTokenSource
from ..infrastructure.http.http_policy import HttpPolicy
from ..infrastructure.ovms.ovms_graph import ovms_plugin_config
from ..infrastructure.ovms.ovms_process import OvmsProcess
from ..infrastructure.ovms.ovms_serving_folder import (
    OvmsExport,
    write_ovms_serving_folder,
)

# The config file OVMS's exporter writes per model (ovms/export_models.sh;
# domino/app.sh serves the same names when it cannot calibrate).
OVMS_CONFIG_FILES: dict[ModelServerPart, str] = {
    "embedder": "config_embedder.json",
    "reranker": "config_reranker.json",
}


class _AnswersReady(Protocol):
    async def ready(self) -> tuple[bool, str]: ...


A = TypeVar("A", bound=_AnswersReady)


class BaseOvmsStage(BaseServedStage, Generic[A]):
    """A stage OVMS serves; ``A`` is the ``api`` adapter that calls it.

    Args:
        settings: The alias, API format, HTTP timeout, OVMS start timeout and
            ``calibration_users``; what the sample is built from.
        client: The HTTP client the adapters share.
        export: Where OVMS's exporter wrote the model, read only.
        serving_dir: The App's own folder each candidate is served from.
        binary: The OVMS executable.

    Example:
        >>> async with stage.serving(StageTuning(2, 20, streams=2)):
        ...     await stage.serve_sample()
    """

    def __init__(
        self,
        settings: RagSettings,
        client: httpx.AsyncClient,
        export: OvmsExport,
        serving_dir: Path,
        binary: str = "ovms",
    ) -> None:
        self._settings, self._client = settings, client
        self._export, self.serving_dir, self._binary = export, serving_dir, binary
        # No retry: a retried request would be timed as one slow request.
        self._policy = HttpPolicy(settings.http_timeout_s, retries=0, backoff_s=0.0)
        self._tokens = RunTokenSource(
            settings.domino_token_url, settings.domino_token_ttl_s
        )
        self._adapter: A | None = None

    @asynccontextmanager
    async def serving(self, tuning: StageTuning) -> AsyncGenerator[None, None]:
        """Write the serving folder for ``tuning`` and run OVMS on it.

        Args:
            tuning: The streams and threads OVMS runs the model with.

        Yields:
            Nothing, once OVMS answers; it is stopped when the context ends.
        """
        config = self.write_folder(tuning)
        process = OvmsProcess(
            config,
            ready=self._answers,
            start_timeout_s=self._settings.ovms_start_timeout_s,
            binary=self._binary,
        )
        async with process:
            yield

    def write_folder(self, tuning: StageTuning) -> Path:
        """Write the serving folder for ``tuning``, as OVMS will be started on it.

        Args:
            tuning: The streams and threads the graph's ``plugin_config`` holds.

        Returns:
            The config file to start OVMS with.

        Example:
            >>> stage.write_folder(StageTuning(2, 20, streams=2))
            PosixPath('/tmp/ovms-serving/config_reranker.json')
        """
        plugin_config = ovms_plugin_config(tuning)
        return write_ovms_serving_folder(self._export, self.serving_dir, plugin_config)

    async def serve_sample(self) -> None:
        """Send ``calibration_users`` sample requests at once; return when all answer.

        Raises:
            RuntimeError: If no server was started: call it inside ``serving``.
        """
        adapter = self._adapter
        if adapter is None:
            raise RuntimeError(
                "serve_sample needs a started server: call it inside serving(tuning)"
            )
        users = self._settings.calibration_users
        await asyncio.gather(*(self._serve_one(adapter) for _ in range(users)))

    def pinned_settings(self, tuning: StageTuning) -> dict[str, int]:
        """Name the two OVMS settings, the same for either model: one App, one model.

        Args:
            tuning: The tuning to pin.

        Returns:
            ``RAG_OVMS_NUM_STREAMS`` and ``RAG_OVMS_THREADS``.

        Example:
            >>> stage.pinned_settings(StageTuning(threads=2, batch_size=1, streams=2))
            {'RAG_OVMS_NUM_STREAMS': 2, 'RAG_OVMS_THREADS': 2}
        """
        return {
            RagSettings.variable_of("ovms_num_streams"): tuning.streams,
            RagSettings.variable_of("ovms_threads"): tuning.threads,
        }

    async def _answers(self, port: int) -> bool:
        # The adapter of this start: readiness is one real minimal request.
        self._adapter = self._adapter_at(f"http://127.0.0.1:{port}")
        answered, _ = await self._adapter.ready()
        return answered

    @abstractmethod
    def _adapter_at(self, origin: str) -> A: ...

    @abstractmethod
    async def _serve_one(self, adapter: A) -> None: ...


class OvmsRerankStage(BaseOvmsStage[ApiReranker]):
    """Rerank on OVMS: the sample ``RerankStage`` uses, sent as one request each.

    Its sample is a question and ``top_k_retrieve`` passages, kept to
    ``top_k_rerank``, as the workflow reranks them.
    """

    stage = "rerank"

    def _adapter_at(self, origin: str) -> ApiReranker:
        settings = self._settings
        endpoint = ApiEndpoint(
            url=origin + MODEL_SERVER_RERANK_PATH,
            api_format=settings.rerank_api_format,
            api_key="",
            model=settings.rerank_api_model,
        )
        return open_api_reranker(
            self._client, endpoint, tokens=self._tokens, policy=self._policy
        )

    @cached_property
    def _sample(self) -> tuple[str, list[ScoredPassage]]:
        sample = calibration_sample(self._settings.top_k_retrieve)
        passages = [
            ScoredPassage(id=f"calibration-{i}", text=text, retrieval_score=0.0)
            for i, text in enumerate(sample.passages)
        ]
        return sample.question, passages

    async def _serve_one(self, adapter: ApiReranker) -> None:
        question, passages = self._sample
        top_n = self._settings.top_k_rerank
        await adapter.rerank(question, passages, top_n=top_n)


class OvmsEmbedStage(BaseOvmsStage[ApiEmbedder]):
    """Embed on OVMS: one question with the query prefix, as ``EmbedStage``'s."""

    stage = "embed"

    def _adapter_at(self, origin: str) -> ApiEmbedder:
        settings = self._settings
        endpoint = ApiEndpoint(
            url=origin + MODEL_SERVER_EMBEDDINGS_PATH,
            api_format=settings.embedding_api_format,
            api_key="",
            model=settings.embedding_api_model,
        )
        return open_api_embedder(
            self._client,
            endpoint,
            tokens=self._tokens,
            policy=self._policy,
            batch_size=1,
        )

    @cached_property
    def _query(self) -> str:
        question = calibration_sample(1).question
        return self._settings.embedding_query_prefix + question

    async def _serve_one(self, adapter: ApiEmbedder) -> None:
        await adapter.embed([self._query])


OvmsStage = OvmsRerankStage | OvmsEmbedStage


def open_ovms_stage(
    part: ModelServerPart,
    settings: RagSettings,
    client: httpx.AsyncClient,
    models_dir: Path,
    serving_dir: Path,
    binary: str = "ovms",
) -> OvmsStage:
    """Return the OVMS stage of ``part``, on the export in ``models_dir``.

    Args:
        part: ``reranker`` or ``embedder``.
        settings: The alias of each model and what its stage reads.
        client: The HTTP client the adapters share.
        models_dir: The folder OVMS's exporter wrote into.
        serving_dir: The App's own folder OVMS serves from.
        binary: The OVMS executable.

    Returns:
        The stage, on the model whose alias the settings name.

    Example:
        >>> open_ovms_stage("embedder", settings, client, models, folder).stage
        'embed'
    """
    if part == "reranker":
        export = OvmsExport(
            models_dir, OVMS_CONFIG_FILES[part], settings.rerank_api_model
        )
        return OvmsRerankStage(settings, client, export, serving_dir, binary)
    alias = settings.embedding_api_model
    export = OvmsExport(models_dir, OVMS_CONFIG_FILES[part], alias)
    return OvmsEmbedStage(settings, client, export, serving_dir, binary)
