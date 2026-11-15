"""Calibration of OVMS: its stream and thread counts, at start-up or on demand.

An OVMS App runs whatever its serving folder's graph says. At start-up
(``qp-ovms-config``, run by ``domino/model_servers/ovms.sh``) each of the two
parameters its settings leave at 0 (``QP_OVMS_NUM_STREAMS``,
``QP_OVMS_THREADS``) is calibrated, and the serving folder is written for the
choice. On demand (``qp-calibrate --ovms-models``) both are searched, pinned or
not, in a temporary folder removed afterwards.

The candidates come from the CPUs counted here (``QP_LOCAL_MODELS__CPUS``, the
Domino tier, the container). Streams: 1, 2, 4, ... up to the CPUs, 1 first
because it is what the export runs and what a spent budget keeps. Threads: the
CPUs, then half of them, down to 1. A request is scored in one batch, so the
tuning's batch size is the request's size: a reranker's
``local.calibration_passages`` passages, or one question.

Before an App's serving folder is written, the export is held to the weight
format the settings ask (``require_asked_weight_format``): the launch script sees
the variable only, and the workflow file may be what states the format.

Example:
    >>> ovms = OvmsCalibration(http, local_models, settings.calibration)
    >>> served = OvmsServedComponent("reranker", reranker_settings, models_dir)
    >>> config = await ovms.serving_config(served, OvmsSettings(), serving_dir)
"""

from __future__ import annotations

import dataclasses
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..config.ovms_settings import OvmsSettings
from ..config.shared_settings import CalibrationSettings
from ..config.workflow_settings import WorkflowSettings
from ..domain.backends import OpenVinoWeightFormat
from ..domain.calibration import CalibrationReport, halvings
from ..infrastructure.calibration.served_stage_calibration import calibrate_served
from ..infrastructure.json_log import log_json
from ..infrastructure.ovms.ovms_serving_folder import (
    OvmsExport,
    recorded_weight_format,
)
from .http_resources import HttpResources
from .local_model_resources import LocalModelResources
from .ovms_stages import (
    OvmsCalls,
    OvmsServedSettings,
    OvmsServer,
    OvmsStage,
    calibration_http_policy,
    open_ovms_stage,
    ovms_config_file,
)

Candidates = tuple[tuple[int, ...], tuple[int, ...]]  # (streams, threads)


def ovms_candidates(cpus: int, *, streams: int, threads: int) -> Candidates:
    """Return the stream and thread counts to try, from the CPUs and two settings.

    A setting above 0 is its one candidate. Stream counts above the first thread
    count are dropped, and thread counts below a pinned stream count, since a
    stream needs a thread.

    Args:
        cpus: Whole CPUs counted here, at least 1.
        streams: ``QP_OVMS_NUM_STREAMS``; 0 = search 1, 2, 4, ... up to ``cpus``.
        threads: ``QP_OVMS_THREADS``; 0 = search ``cpus``, half of it, ... 1.

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


@dataclass(frozen=True)
class OvmsServedComponent:
    """One component of the workflow file, as OVMS's exporter wrote it.

    Example:
        >>> OvmsServedComponent("reranker", reranker_settings, models_dir).export
        OvmsExport(models_dir=..., config_file='config_reranker.json', alias='...')
    """

    name: str  # the component's name in the workflow file
    settings: OvmsServedSettings  # its alias, and what its sample is built from
    models_dir: Path  # the folder OVMS's exporter wrote into, read only

    @property
    def export(self) -> OvmsExport:
        """Where the exporter wrote it: its config file and its alias's folder."""
        return OvmsExport(
            self.models_dir, ovms_config_file(self.name), self.settings.alias
        )


def require_asked_weight_format(
    served: OvmsServedComponent, asked: OpenVinoWeightFormat
) -> None:
    """Refuse an OVMS export of another weight format than the settings ask.

    OVMS would serve it all the same, and the App would then run other weights
    than the in-process models and than its settings say. A folder that holds no
    export of the component is not judged here: building the serving folder
    says that it is missing.

    Args:
        served: The component, and the folder OVMS's exporter wrote into.
        asked: The settings' ``local_models.openvino_weight_format``.

    Raises:
        ValueError: If the folder holds another weight format; the message names
            the folder, both formats and both ways out.

    Example:
        >>> require_asked_weight_format(served, "fp32")
    """
    models_dir = served.models_dir
    if not (models_dir / served.export.config_file).is_file():
        return
    found = recorded_weight_format(models_dir)
    if found == asked:
        return
    variable = WorkflowSettings.variable_of("local_models", "openvino_weight_format")
    raise ValueError(
        f"{models_dir} holds an OVMS export at weight format {found}, but {asked} "
        f"was asked (local_models.openvino_weight_format of the workflow file, "
        f"{variable}); set {variable}={found} to serve that export, or point "
        f"QP_OVMS_MODELS_DIR at another folder, where {asked} is exported on "
        "first start"
    )


@dataclass(frozen=True)
class OvmsAppCalibration:
    """One component's OVMS calibration on this machine, and what pins it.

    The pinned variables are an OVMS App's own (``QP_OVMS_*``), no value of the
    workflow file: one OVMS App serves one component, so they go in that App's
    block of ``deploy.yaml`` (``apps.<component>.settings``).

    Example:
        >>> calibration = await composition.ovms_calibration("reranker", ovms, models)
        >>> calibration.pinned
        {'QP_OVMS_NUM_STREAMS': 2, 'QP_OVMS_THREADS': 8}
    """

    component: str  # the component's name in the workflow file
    report: CalibrationReport
    cpus: float  # the CPUs counted here
    pinned: dict[str, int]  # the App's variables that pin the choice, and values
    runtime: str = "ovms"  # what ran the model


class OvmsCalibration:
    """Calibrates OVMS for one component, and writes the folder it serves from.

    Args:
        http: The shared HTTP client the requests go through, and how patiently
            each component is called.
        local_models: Counts the CPUs, and holds the calibration budget and
            rounds.
        calibration: The ``calibration`` block, for the budget a sample request
            may wait and the requests in flight.
        binary: The OVMS executable.

    Example:
        >>> await OvmsCalibration(http, local_models, settings.calibration
        ...     ).calibration(served, OvmsSettings())
    """

    def __init__(
        self,
        http: HttpResources,
        local_models: LocalModelResources,
        calibration: CalibrationSettings,
        binary: str = "ovms",
    ) -> None:
        self._http, self._local_models = http, local_models
        self._calibration, self._binary = calibration, binary

    async def serving_config(
        self, served: OvmsServedComponent, ovms: OvmsSettings, serving_dir: Path
    ) -> Path:
        """Calibrate what the settings leave at 0, then write the serving folder.

        Each candidate is served from ``serving_dir`` itself; nothing is started
        when both parameters are pinned. One ``ovms_serving_config`` JSON line
        logs the result.

        Args:
            served: The component, and the folder OVMS's exporter wrote into.
            ovms: The App's pinned stream and thread counts (0 = calibrated)
                and its start timeout.
            serving_dir: The App's own folder OVMS will serve from.

        Returns:
            The config file to start OVMS with.

        Example:
            >>> await ovms_calibration.serving_config(served, ovms, Path("/tmp/s"))
            PosixPath('/tmp/s/config_reranker.json')
        """
        stage = await self._stage(served, ovms, serving_dir)
        streams, threads = ovms_candidates(
            await self._cpus(), streams=ovms.num_streams, threads=ovms.threads
        )
        report = await self._search(stage, streams, threads)
        config = stage.write_folder(report.chosen)
        log_json(
            event="ovms_serving_config",
            component=served.name,
            streams=report.chosen.streams,
            threads=report.chosen.threads,
            calibrated=bool(report.trials),
            config=str(config),
        )
        return config

    async def calibration(
        self, served: OvmsServedComponent, ovms: OvmsSettings
    ) -> OvmsAppCalibration:
        """Calibrate OVMS on one component here, searching both parameters.

        A pinned stream or thread count is searched too: revisiting it is the
        point.

        Args:
            served: The component, and the folder OVMS's exporter wrote into.
            ovms: The start timeout; its pinned counts are not read.

        Returns:
            The report, the CPUs it ran on, ``ovms`` as the runtime, and the
            App's variables that pin the choice.

        Example:
            >>> (await ovms_calibration.calibration(served, ovms)).pinned
            {'QP_OVMS_NUM_STREAMS': 2, 'QP_OVMS_THREADS': 2}
        """
        machine = await self._local_models.unpinned_budget()
        streams, threads = ovms_candidates(await self._cpus(), streams=0, threads=0)
        with tempfile.TemporaryDirectory(prefix="qp-ovms-calibration-") as folder:
            stage = await self._stage(served, ovms, Path(folder))
            report = await self._search(stage, streams, threads)
        pinned = stage.pinned_settings(report.chosen)
        return OvmsAppCalibration(served.name, report, cpus=machine.cpus, pinned=pinned)

    async def _search(
        self, stage: OvmsStage, streams: tuple[int, ...], threads: tuple[int, ...]
    ) -> CalibrationReport:
        return await calibrate_served(
            stage,
            streams=streams,
            threads=threads,
            # OVMS scores a request in one batch: the batch is the request.
            batch_size=stage.request_size,
            requests=self._calibration.users,
            limits=self._local_models.calibration_limits,
        )

    async def _stage(
        self, served: OvmsServedComponent, ovms: OvmsSettings, serving_dir: Path
    ) -> OvmsStage:
        http, component = self._http, served.settings
        # One readiness request, alone and without retry, under the timeout the
        # component is called with; the sample's wait up to the budget.
        ready = dataclasses.replace(http.policy_of(component), retries=0, backoff_s=0.0)
        calls = OvmsCalls(
            await http.client(),
            http.run_tokens,
            ready=ready,
            sample=calibration_http_policy(self._calibration),
            users=self._calibration.users,
        )
        server = OvmsServer(
            served.export, serving_dir, ovms.start_timeout_s, self._binary
        )
        return open_ovms_stage(component, calls, server)

    async def _cpus(self) -> int:
        # OVMS runs on whole CPUs; a fractional quota still has at least one.
        machine = await self._local_models.unpinned_budget()
        return max(1, int(machine.cpus))
