"""``qp-calibrate``: calibrate the CPU-bound stages here, print what pins them.

An App calibrates each in-process model parameter left at 0 when it starts, and
every start and every replica does it again. This command does it once, on a
machine of the App's hardware tier that holds the models, and prints what it
found as ``.env`` lines: for each component a ``#`` comment, a second one with
its entry for ``overrides.components`` of ``deploy.yaml``, then the
environment paths of its ``local`` block that pin the choice, as
``KEY=VALUE`` lines. Pinned under the overrides of ``deploy.yaml`` or in
``.env``, they make an App start without calibrating, and every replica run
the same values.

It calibrates the component ``--only`` names, else every embedder and reranker
of the workflow file. Each model loads as an App would load it (``QP_*``
overrides honoured), whatever its Backend says, and every parameter
is searched, pinned or not.
With ``--ovms-models DIR`` it calibrates OpenVINO Model Server instead, on the
export in DIR: its stream and thread counts, one ``ovms`` start per candidate.
A component then prints the OVMS App's own two variables
(``QP_OVMS_NUM_STREAMS``, ``QP_OVMS_THREADS``), and its second comment gives
their place in ``deploy.yaml``: that App's block, since one OVMS App serves one
component.
The logs, ``stage_calibrated`` included, go to standard error, so standard
output can be appended to ``.env`` as it is.

Example:
    $ QP_CALIBRATION__ROUNDS=3 qp-calibrate --only reranker >> .env
    $ qp-calibrate --ovms-models /mnt/data/ovms-models --only reranker
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import yaml

from ..composition.calibrated_stages import ComponentCalibration
from ..composition.composition_root import CompositionRoot
from ..composition.ovms_calibration import OvmsAppCalibration
from ..composition.registered_names import check_registered_names
from ..config.apps.ovms_settings import OvmsSettings
from ..config.workflow_settings import WorkflowSettings
from ..domain.errors import RagDependencyError
from ..domain.runtime.calibration import CalibrationReport
from ..infrastructure.local_models.model_provisioning import ModelProvisioningError
from ..infrastructure.local_models.openvino_export import OpenVinoUnavailableError
from ..infrastructure.one_line_failure import one_line_failure
from ..infrastructure.ovms.ovms_process import OvmsStartError

PROG = "qp-calibrate"
# The component types that run a model in-process, so can be calibrated; they
# are the ones OVMS serves on an export, and calibrates (it pulls an LLM itself).
CALIBRATED_TYPES = ("embedder", "reranker")
# Failures an operator can act on (a bad setting, a model that cannot be found,
# downloaded or exported, a missing package or ovms, an ovms that would not
# start, a dependency that would not open); anything else is a bug and keeps its
# traceback.
EXPECTED_ERRORS = (
    ModelProvisioningError,
    OpenVinoUnavailableError,
    OvmsStartError,
    RagDependencyError,
    OSError,
    ValueError,
)
# One component's calibration: of its in-process model, or of OVMS serving it.
Calibration = ComponentCalibration | OvmsAppCalibration


class OvmsTarget(NamedTuple):
    """OVMS's export to calibrate OVMS on, and the OVMS App's own settings.

    Example:
        >>> OvmsTarget(Path("/mnt/data/ovms-models"), OvmsSettings())
    """

    models_dir: Path  # the folder OVMS's exporter wrote into; never written
    settings: OvmsSettings  # for the start timeout: both counts are searched


def env_lines(calibration: Calibration) -> list[str]:
    """Return a component's lines for a ``.env`` file: two comments, then its settings.

    Args:
        calibration: One component's calibration.

    Returns:
        The ``#`` comment, the ``#`` line of its place in ``deploy.yaml`` (its
        entry under ``overrides.components``, or under its App's ``settings``
        for OVMS), then one ``KEY=VALUE`` line per pinned setting.

    Example:
        >>> env_lines(calibration)[1:]
        ['# deploy.yaml, under overrides.components: reranker: {local: ...}',
         'QP_COMPONENTS__RERANKER__LOCAL__THREADS=2',
         'QP_COMPONENTS__RERANKER__LOCAL__BATCH_SIZE=2']
    """
    report = calibration.report
    where = f"timed on {calibration.cpus:g} CPUs under {calibration.runtime}"
    comment = (
        f"# {calibration.component} ({report.stage}): {where}, "
        f"{_choice(report)}; {_search(report)}"
    )
    pinned = (f"{name}={value}" for name, value in calibration.pinned.items())
    return [comment, _deploy_line(calibration), *pinned]


def _deploy_line(calibration: Calibration) -> str:
    # One line of deploy.yaml; a comment, so the output still appends to .env
    # as it is. An in-process model's settings are values of the workflow file,
    # under the components of the overrides, in its shape. OVMS's are its App's
    # own variables, in that App's block: one OVMS App serves one component.
    if isinstance(calibration, OvmsAppCalibration):
        where = f"apps.{calibration.component}.settings"
        return _flow_line(where, calibration.pinned)
    entry = {calibration.component: {"local": calibration.local}}
    return _flow_line("overrides.components", entry)


def _flow_line(where: str, entry: Mapping[str, object]) -> str:
    flow = yaml.safe_dump(
        dict(entry), default_flow_style=True, sort_keys=False, width=sys.maxsize
    ).strip()
    inner = flow.removeprefix("{").removesuffix("}")
    return f"# deploy.yaml, under {where}: {inner}"


def _choice(report: CalibrationReport) -> str:
    seconds = report.chosen_seconds
    if seconds is None and report.cut_short:  # the first trial was abandoned
        return "the first candidate, not timed"
    if seconds is None:  # one candidate per parameter: nothing to compare
        return "the only candidate, not timed"
    if report.requests > 1:
        return f"{seconds:.3f} s for {report.requests} requests at once"
    return f"{seconds:.3f} s per request"


def _search(report: CalibrationReport) -> str:
    tried = len(report.trials)
    noun = "candidate" if tried == 1 else "candidates"
    text = f"tried {tried} {noun} in {report.seconds:.1f} s"
    if report.cut_short:
        budget = WorkflowSettings.variable_of("calibration", "budget_s")
        text += f", cut short by {budget}={report.budget_s:g}"
    return text


def calibrated_components(settings: WorkflowSettings) -> list[str]:
    """Return the components that run a model in-process: embedders and rerankers.

    Args:
        settings: The settings whose workflow file declares the components.

    Returns:
        Their names, in the file's order.

    Example:
        >>> calibrated_components(WorkflowSettings())
        ['embedder', 'reranker']
    """
    return [n for n, c in settings.components.items() if c.type in CALIBRATED_TYPES]


async def calibrate_components(
    settings: WorkflowSettings, names: Sequence[str], ovms: OvmsTarget | None = None
) -> int:
    """Calibrate each component in turn, printing its lines as soon as it is done.

    Args:
        settings: The settings the components are opened from.
        names: Which components to calibrate, in order.
        ovms: OVMS's export to calibrate OVMS on, and its settings; None for
            the in-process models.

    Returns:
        0 once every component is printed; 1 after an expected failure,
        reported as one line on stderr.

    Example:
        >>> await calibrate_components(WorkflowSettings(), ["reranker"])  # its lines
        0
    """
    async with CompositionRoot(settings) as composition:
        return await _print_each(composition, names, ovms)


async def _print_each(
    composition: CompositionRoot, names: Sequence[str], ovms: OvmsTarget | None
) -> int:
    # Each component is printed as soon as it is done, so a later failure keeps it.
    for name in names:
        try:
            calibration = await _calibration(composition, name, ovms)
        except EXPECTED_ERRORS as exc:
            return one_line_failure(PROG, f"cannot calibrate {name}", exc)
        print("\n".join(env_lines(calibration)), flush=True)
    return 0


async def _calibration(
    composition: CompositionRoot, name: str, ovms: OvmsTarget | None
) -> Calibration:
    if ovms is None:
        return await composition.calibration(name)
    return await composition.ovms_calibration(name, ovms.settings, ovms.models_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``qp-calibrate``; ``QP_*`` overrides are honoured, as everywhere.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status: 0 when every requested component was calibrated, else 1;
        a name that is no embedder or reranker exits as a usage error.

    Example:
        >>> main(["--only", "reranker"])  # prints the reranker's .env lines
        0
    """
    parser = _parser()
    args = parser.parse_args(argv)
    # The Apps' log format, on stderr: stdout holds only the .env lines.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        settings = WorkflowSettings()
        check_registered_names(settings)
        ovms = _ovms_target(args.ovms_models)
    except ValueError as exc:  # pydantic's ValidationError is one
        return one_line_failure(PROG, "cannot read the settings", exc)
    names = calibrated_components(settings)
    if args.only and args.only not in names:
        parser.error(
            f"--only {args.only!r}: expected an embedder or a reranker of the "
            f"workflow file: {', '.join(names)}"
        )
    return asyncio.run(
        calibrate_components(settings, [args.only] if args.only else names, ovms)
    )


def _ovms_target(models_dir: Path | None) -> OvmsTarget | None:
    # The OVMS App's own settings are read only when OVMS is what is calibrated.
    return None if models_dir is None else OvmsTarget(models_dir, OvmsSettings())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Calibrate the in-process embedders and rerankers on this "
        "machine and print the environment paths that pin the result, as .env "
        "lines, each component's after a comment that gives its entry under "
        "overrides.components of deploy.yaml.",
    )
    parser.add_argument(
        "--only",
        metavar="NAME",
        help="calibrate just this component, such as reranker",
    )
    parser.add_argument(
        "--ovms-models",
        type=Path,
        metavar="DIR",
        help="calibrate OpenVINO Model Server on the export in DIR (needs ovms on "
        "PATH) instead of the in-process models; it prints QP_OVMS_NUM_STREAMS "
        "and QP_OVMS_THREADS, pinned per App (one OVMS App serves one component)",
    )
    return parser
