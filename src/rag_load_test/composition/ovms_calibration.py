"""Calibration of OVMS: its stream and thread counts, at start-up or on demand.

An OVMS App runs whatever its serving folder's graph says. At start-up
(``rag-ovms-config``, run by ``domino/app.sh``) each of the two parameters the
settings leave at 0 is calibrated, and the serving folder is written for the
choice. On demand (``rag-calibrate --ovms-models``) both are searched, pinned or
not, in a temporary folder removed afterwards.

The candidates come from the CPUs counted here (``RAG_MODEL_CPUS``, the Domino
tier, the container). Streams: 1, 2, 4, ... up to the CPUs, 1 first because it
is what the export runs and what a spent budget keeps. Threads: the CPUs, then
half of them, down to 1. A request is scored in one batch, so the tuning's batch
size is the request's size: ``RAG_TOP_K_RETRIEVE`` passages, or one question.

Example:
    >>> ovms = OvmsCalibration(settings, http, local_models)
    >>> config = await ovms.serving_config("reranker", models_dir, serving_dir)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..config.rag_settings import RagSettings
from ..domain.calibration import CalibrationReport, halvings
from ..domain.roles import ModelServerPart
from ..infrastructure.calibration.served_stage_calibration import calibrate_served
from ..infrastructure.json_log import log_json
from .calibrated_stages import ComponentCalibration, calibration_limits
from .http_resources import HttpResources
from .local_model_resources import LocalModelResources
from .ovms_stages import OvmsStage, open_ovms_stage

Candidates = tuple[tuple[int, ...], tuple[int, ...]]  # (streams, threads)


def ovms_candidates(cpus: int, *, streams: int, threads: int) -> Candidates:
    """Return the stream and thread counts to try, from the CPUs and two settings.

    A setting above 0 is its one candidate. Stream counts above the first thread
    count are dropped, and thread counts below a pinned stream count, since a
    stream needs a thread.

    Args:
        cpus: Whole CPUs counted here, at least 1.
        streams: ``ovms_num_streams``; 0 = search 1, 2, 4, ... up to ``cpus``.
        threads: ``ovms_threads``; 0 = search ``cpus``, half of it, ... 1.

    Returns:
        The stream counts, 1 first, and the thread counts, the largest first.

    Example:
        >>> ovms_candidates(4, streams=0, threads=0)
        ((1, 2, 4), (4, 2, 1))
    """
    thread_counts = (threads,) if threads else halvings(cpus)
    if streams:
        fitting = tuple(count for count in thread_counts if count >= streams)
        return (streams,), fitting or (streams,)
    ups = tuple(reversed(halvings(cpus)))
    return tuple(c for c in ups if c <= thread_counts[0]), thread_counts


class OvmsCalibration:
    """Calibrates OVMS for one model, and writes the folder it serves from.

    Args:
        settings: The OVMS settings, the calibration budget, rounds and users,
            and what each model's sample and alias come from.
        http: The shared HTTP client the requests go through.
        local_models: Counts the CPUs.
        binary: The OVMS executable.

    Example:
        >>> await OvmsCalibration(settings, http, local_models).calibration(
        ...     "embedder", Path("/mnt/data/ovms-models"))
    """

    def __init__(
        self,
        settings: RagSettings,
        http: HttpResources,
        local_models: LocalModelResources,
        binary: str = "ovms",
    ) -> None:
        self._settings, self._http = settings, http
        self._local_models, self._binary = local_models, binary

    async def serving_config(
        self, part: ModelServerPart, models_dir: Path, serving_dir: Path
    ) -> Path:
        """Calibrate what the settings leave at 0, then write the serving folder.

        Each candidate is served from ``serving_dir`` itself; nothing is started
        when both parameters are pinned. One ``ovms_serving_config`` JSON line
        logs the result.

        Args:
            part: ``reranker`` or ``embedder``.
            models_dir: The folder OVMS's exporter wrote into, read only.
            serving_dir: The App's own folder OVMS will serve from.

        Returns:
            The config file to start OVMS with.

        Example:
            >>> await ovms.serving_config("reranker", models, Path("/tmp/s"))
            PosixPath('/tmp/s/config_reranker.json')
        """
        settings = self._settings
        stage = await self._stage(part, models_dir, serving_dir)
        cpus = await self._cpus()
        streams, threads = ovms_candidates(
            cpus, streams=settings.ovms_num_streams, threads=settings.ovms_threads
        )
        report = await self._search(stage, part, streams, threads)
        config = stage.write_folder(report.chosen)
        log_json(
            event="ovms_serving_config",
            part=part,
            streams=report.chosen.streams,
            threads=report.chosen.threads,
            calibrated=bool(report.trials),
            config=str(config),
        )
        return config

    async def calibration(
        self, part: ModelServerPart, models_dir: Path
    ) -> ComponentCalibration:
        """Calibrate OVMS on one model here, searching both parameters even if pinned.

        Args:
            part: ``reranker`` or ``embedder``.
            models_dir: The folder OVMS's exporter wrote into, read only.

        Returns:
            The report, the CPUs it ran on, ``ovms`` as the runtime, and the
            settings that pin the choice.

        Example:
            >>> (await ovms.calibration("reranker", models)).pinned
            {'RAG_OVMS_NUM_STREAMS': 2, 'RAG_OVMS_THREADS': 2}
        """
        machine = await self._local_models.unpinned_budget()
        streams, threads = ovms_candidates(await self._cpus(), streams=0, threads=0)
        with tempfile.TemporaryDirectory(prefix="rag-ovms-calibration-") as folder:
            stage = await self._stage(part, models_dir, Path(folder))
            report = await self._search(stage, part, streams, threads)
        pinned = stage.pinned_settings(report.chosen)
        return ComponentCalibration(
            report, cpus=machine.cpus, runtime="ovms", pinned=pinned
        )

    async def _search(
        self,
        stage: OvmsStage,
        part: ModelServerPart,
        streams: tuple[int, ...],
        threads: tuple[int, ...],
    ) -> CalibrationReport:
        settings = self._settings
        # OVMS scores a request in one batch: the batch is the request.
        batch_size = settings.top_k_retrieve if part == "reranker" else 1
        return await calibrate_served(
            stage,
            streams=streams,
            threads=threads,
            batch_size=batch_size,
            requests=settings.calibration_users,
            limits=calibration_limits(settings),
        )

    async def _stage(
        self, part: ModelServerPart, models_dir: Path, serving_dir: Path
    ) -> OvmsStage:
        client = await self._http.client()
        return open_ovms_stage(
            part, self._settings, client, models_dir, serving_dir, self._binary
        )

    async def _cpus(self) -> int:
        # OVMS runs on whole CPUs; a fractional quota still has at least one.
        machine = await self._local_models.unpinned_budget()
        return max(1, int(machine.cpus))
