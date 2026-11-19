"""The stages OVMS serves, as calibration times them: rerank and embed.

Each candidate tuning gets a serving folder written for it and one ``ovms``
start (``infrastructure/ovms/``); the sample's requests are then sent at once,
``calibration.users`` of them, through the same ``api`` adapters the workflow
app calls a served component with, on OVMS's /v3 routes, in the standard format
of the component's type and with its alias as model.
``infrastructure/calibration/served_stage_calibration.py`` times them. The
sample is the one the component's in-process calibration serves: a reranker's
``local.calibration_passages`` passages, all kept, or one question an embedder
embeds as a query.

Example:
    >>> stage = open_ovms_stage(reranker_settings, calls, server)
    >>> report = await calibrate_served(stage, streams=(1, 2), threads=(2, 1), ...)
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Generic, Protocol, TypeVar

import httpx

from ..adapters.embedder.api.api_embedder import ApiEmbedder, open_api_embedder
from ..adapters.reranker.api.api_reranker import ApiReranker, open_api_reranker
from ..config.apps.ovms_settings import OvmsSettings
from ..config.components.component_settings import EmbedderSettings, RerankerSettings
from ..config.shared_settings import CalibrationSettings
from ..domain.endpoints.api_endpoint import (
    MODEL_SERVER_EMBEDDINGS_PATH,
    MODEL_SERVER_RERANK_PATH,
    ApiEndpoint,
)
from ..domain.endpoints.standard_formats import STANDARD_API_FORMATS
from ..domain.retrieval.passages import ScoredPassage
from ..domain.runtime.calibration import StageTuning, calibration_sample
from ..infrastructure.calibration.served_stage_calibration import BaseServedStage
from ..infrastructure.endpoint.open_endpoint_client import RunTokenSource
from ..infrastructure.http.http_policy import HttpPolicy
from ..infrastructure.ovms.ovms_graph import ovms_plugin_config
from ..infrastructure.ovms.ovms_process import OvmsProcess
from ..infrastructure.ovms.ovms_serving_folder import (
    OvmsExport,
    write_ovms_serving_folder,
)

# A component OVMS can serve: the types its exporter exports.
OvmsServedSettings = EmbedderSettings | RerankerSettings


def ovms_config_file(component: str) -> str:
    """Return the name of the config file OVMS's exporter writes for a component.

    ``ovms/export_models.sh`` writes it, and ``domino/model_servers/ovms.sh``
    serves the same name when it cannot calibrate.

    Args:
        component: The component's name in the workflow file.

    Returns:
        ``config_<component>.json``.

    Example:
        >>> ovms_config_file("reranker")
        'config_reranker.json'
    """
    return f"config_{component}.json"


def calibration_http_policy(calibration: CalibrationSettings) -> HttpPolicy:
    """Return how patiently the calibration adapters wait for OVMS's answers.

    The timeout is the calibration budget, not the component's HTTP timeout: a
    request legitimately waits behind the others in flight (GCP sitting of
    2026-11-15, 16 requests on one stream at about 2 s each: the last waited
    32 s, past the 30 s HTTP timeout, and failed the App). The budget is what
    bounds a trial (``calibrate_served``). No retry: a retried request would be
    timed as one slow request.

    Args:
        calibration: The ``calibration`` block, for ``budget_s``.

    Returns:
        The policy of the sample's requests.

    Example:
        >>> calibration_http_policy(CalibrationSettings()).timeout_s
        90.0
    """
    return HttpPolicy(calibration.budget_s, retries=0, backoff_s=0.0)


@dataclass(frozen=True)
class OvmsCalls:
    """How the candidates of one OVMS calibration are called.

    Example:
        >>> calls = OvmsCalls(client, tokens, ready=ready_policy,
        ...                   sample=calibration_http_policy(calibration), users=4)
    """

    client: httpx.AsyncClient  # the HTTP client the adapters share
    tokens: RunTokenSource  # never asked: OVMS on 127.0.0.1 takes no key
    # The readiness request while OVMS starts is one, alone: the component's own
    # HTTP timeout bounds it. The sample's requests wait up to the budget.
    ready: HttpPolicy
    sample: HttpPolicy
    users: int  # sample requests in flight at once (calibration.users)


@dataclass(frozen=True)
class OvmsServer:
    """What one OVMS calibration starts its candidates on.

    Example:
        >>> server = OvmsServer(export, Path("/tmp/serving"), start_timeout_s=120.0)
    """

    export: OvmsExport  # where OVMS's exporter wrote the model, read only
    serving_dir: Path  # the App's own folder each candidate is served from
    start_timeout_s: float  # how long one start may take (QP_OVMS_START_TIMEOUT_S)
    binary: str = "ovms"  # the OVMS executable


class _AnswersReady(Protocol):
    async def ready(self) -> tuple[bool, str]: ...


C = TypeVar("C", EmbedderSettings, RerankerSettings)
A = TypeVar("A", bound=_AnswersReady)


class BaseOvmsStage(BaseServedStage, Generic[C, A]):
    """A stage OVMS serves; ``A`` is the ``api`` adapter that calls it.

    Args:
        component: The component OVMS serves: its alias, its type's standard
            API format, and what its sample is built from.
        calls: The HTTP client, the policies of the readiness and sample
            requests, and how many sample requests are in flight.
        server: The export, the App's own serving folder, the start timeout
            and the OVMS executable.

    Example:
        >>> async with stage.serving(StageTuning(2, 20, streams=2)):
        ...     await stage.serve_sample()
    """

    def __init__(self, component: C, calls: OvmsCalls, server: OvmsServer) -> None:
        self._component: C = component
        self._calls, self._server = calls, server
        self._adapter: A | None = None

    @property
    @abstractmethod
    def request_size(self) -> int:
        """The texts one sample request holds: OVMS scores a request in one batch."""

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
            start_timeout_s=self._server.start_timeout_s,
            binary=self._server.binary,
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
        server = self._server
        plugin_config = ovms_plugin_config(tuning)
        return write_ovms_serving_folder(
            server.export, server.serving_dir, plugin_config
        )

    async def serve_sample(self) -> None:
        """Send ``calibration.users`` sample requests at once; return when all answer.

        Raises:
            RuntimeError: If no server was started: call it inside ``serving``.
        """
        adapter = self._adapter
        if adapter is None:
            raise RuntimeError(
                "serve_sample needs a started server: call it inside serving(tuning)"
            )
        users = self._calls.users
        await asyncio.gather(*(self._serve_one(adapter) for _ in range(users)))

    def pinned_settings(self, tuning: StageTuning) -> dict[str, int]:
        """Name the two OVMS variables, the same for any component: one App, one model.

        Args:
            tuning: The tuning to pin.

        Returns:
            ``QP_OVMS_NUM_STREAMS`` and ``QP_OVMS_THREADS``.

        Example:
            >>> stage.pinned_settings(StageTuning(threads=2, batch_size=1, streams=2))
            {'QP_OVMS_NUM_STREAMS': 2, 'QP_OVMS_THREADS': 2}
        """
        return {
            OvmsSettings.variable_of("num_streams"): tuning.streams,
            OvmsSettings.variable_of("threads"): tuning.threads,
        }

    def _endpoint(self, origin: str, route: str) -> ApiEndpoint:
        # A model server of ours speaks the standard format of the component's
        # type and serves the model under its alias; OVMS here takes no key.
        component = self._component
        return ApiEndpoint(
            url=origin + route,
            api_format=STANDARD_API_FORMATS[component.type],
            api_key="",
            model=component.alias,
        )

    async def _answers(self, port: int) -> bool:
        # The adapters of this start: readiness is one real minimal request.
        origin = f"http://127.0.0.1:{port}"
        self._adapter = self._adapter_at(origin, self._calls.sample)
        answered, _ = await self._adapter_at(origin, self._calls.ready).ready()
        return answered

    @abstractmethod
    def _adapter_at(self, origin: str, policy: HttpPolicy) -> A: ...

    @abstractmethod
    async def _serve_one(self, adapter: A) -> None: ...


class OvmsRerankStage(BaseOvmsStage[RerankerSettings, ApiReranker]):
    """Rerank on OVMS: the sample ``CalibratedRerankStage`` serves, one request each.

    Its sample is a question and ``local.calibration_passages`` passages, all
    kept: the model scores every candidate whatever it keeps.
    """

    stage = "rerank"

    @property
    def request_size(self) -> int:
        """The passages of the sample request."""
        return self._component.local.calibration_passages

    def _adapter_at(self, origin: str, policy: HttpPolicy) -> ApiReranker:
        calls = self._calls
        endpoint = self._endpoint(origin, MODEL_SERVER_RERANK_PATH)
        return open_api_reranker(
            calls.client, endpoint, tokens=calls.tokens, policy=policy
        )

    @cached_property
    def _sample(self) -> tuple[str, list[ScoredPassage]]:
        sample = calibration_sample(self.request_size)
        passages = [
            ScoredPassage(id=f"calibration-{i}", text=text, retrieval_score=0.0)
            for i, text in enumerate(sample.passages)
        ]
        return sample.question, passages

    async def _serve_one(self, adapter: ApiReranker) -> None:
        question, passages = self._sample
        await adapter.rerank(question, passages, top_n=len(passages))


class OvmsEmbedStage(BaseOvmsStage[EmbedderSettings, ApiEmbedder]):
    """Embed on OVMS: one question embedded as a query, as an App embeds one."""

    stage = "embed"

    @property
    def request_size(self) -> int:
        """One text: a query."""
        return 1

    def _adapter_at(self, origin: str, policy: HttpPolicy) -> ApiEmbedder:
        calls = self._calls
        endpoint = self._endpoint(origin, MODEL_SERVER_EMBEDDINGS_PATH)
        return open_api_embedder(
            calls.client,
            endpoint,
            tokens=calls.tokens,
            policy=policy,
            batch_size=1,
            query_prefix=self._component.query_prefix,
        )

    @cached_property
    def _question(self) -> str:
        return calibration_sample(1).question

    async def _serve_one(self, adapter: ApiEmbedder) -> None:
        await adapter.embed_query(self._question)


OvmsStage = OvmsRerankStage | OvmsEmbedStage


def open_ovms_stage(
    component: OvmsServedSettings, calls: OvmsCalls, server: OvmsServer
) -> OvmsStage:
    """Return the OVMS stage of ``component``, on the export ``server`` names.

    Args:
        component: The embedder or reranker OVMS serves.
        calls: How its candidates are called.
        server: What its candidates are started on.

    Returns:
        The rerank stage of a reranker, the embed stage of an embedder.

    Example:
        >>> open_ovms_stage(embedder_settings, calls, server).stage
        'embed'
    """
    if isinstance(component, RerankerSettings):
        return OvmsRerankStage(component, calls, server)
    return OvmsEmbedStage(component, calls, server)
