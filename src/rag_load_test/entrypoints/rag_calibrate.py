"""``rag-calibrate``: calibrate the CPU-bound stages here, print what pins them.

An App calibrates each in-process model parameter left at 0 when it starts, and
every start and every replica does it again. This command does it once, on a
machine of the App's hardware tier that holds the models, and prints what it
found as ``.env`` lines: one ``#`` comment per stage, then its ``KEY=VALUE``
lines. Pinned in ``deploy.yaml``'s ``settings`` or ``.env``, they make an App
start without calibrating, and every replica run the same values.

Each model loads as an App would load it (``RAG_*`` overrides honoured), whatever
the topology or Backend says, and every parameter is searched, pinned or not.
With ``--ovms-models DIR`` it calibrates OpenVINO Model Server instead, on the
export in DIR: its stream and thread counts, one ``ovms`` start per candidate.
The logs, ``stage_calibrated`` included, go to standard error, so standard
output can be appended to ``.env`` as it is.

Example:
    $ RAG_CALIBRATION_ROUNDS=3 rag-calibrate --only reranker >> .env
    $ rag-calibrate --ovms-models /mnt/data/ovms-models --only reranker
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from ..composition.calibrated_stages import ComponentCalibration
from ..composition.composition_root import CompositionRoot
from ..config.rag_settings import RagSettings
from ..domain.calibration import CalibrationReport
from ..domain.errors import RagDependencyError
from ..domain.roles import ModelServerPart
from ..infrastructure.local_models.model_provisioning import ModelProvisioningError
from ..infrastructure.local_models.openvino_export import OpenVinoUnavailableError
from ..infrastructure.one_line_failure import one_line_failure
from ..infrastructure.ovms.ovms_process import OvmsStartError

PROG = "rag-calibrate"
PARTS: tuple[ModelServerPart, ...] = get_args(ModelServerPart)  # embedder, reranker
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


def env_lines(calibration: ComponentCalibration) -> list[str]:
    """Return a stage's lines for a ``.env`` file: one comment, then its settings.

    Args:
        calibration: One component's calibration.

    Returns:
        The ``#`` comment, then one ``KEY=VALUE`` line per pinned setting.

    Example:
        >>> env_lines(calibration)[1:]
        ['RAG_RERANK_THREADS=2', 'RAG_RERANK_BATCH_SIZE=2']
    """
    report = calibration.report
    where = f"timed on {calibration.cpus:g} CPUs under {calibration.runtime}"
    comment = f"# {report.stage}: {where}, {_choice(report)}; {_search(report)}"
    return [comment, *(f"{name}={value}" for name, value in calibration.pinned.items())]


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
        text += f", cut short by RAG_CALIBRATION_BUDGET_S={report.budget_s:g}"
    return text


async def calibrate_parts(
    parts: Sequence[ModelServerPart], ovms_models: Path | None = None
) -> int:
    """Calibrate each part in turn, printing its lines as soon as it is done.

    Args:
        parts: Which components to calibrate, in order.
        ovms_models: OVMS's export to calibrate OVMS on; None for the
            in-process models.

    Returns:
        0 once every part is printed; 1 after an expected failure, reported as
        one line on stderr.

    Example:
        >>> await calibrate_parts(["reranker"])  # prints the reranker's lines
        0
    """
    try:
        settings = RagSettings()
    except ValueError as exc:  # pydantic's ValidationError is one
        return one_line_failure(PROG, "cannot read the settings", exc)
    async with CompositionRoot(settings) as composition:
        return await _print_each(composition, parts, ovms_models)


async def _print_each(
    composition: CompositionRoot,
    parts: Sequence[ModelServerPart],
    ovms_models: Path | None,
) -> int:
    # Each part is printed as soon as it is done, so a later failure keeps it.
    for part in parts:
        try:
            calibration = await _calibration(composition, part, ovms_models)
        except EXPECTED_ERRORS as exc:
            return one_line_failure(PROG, f"cannot calibrate {part}", exc)
        print("\n".join(env_lines(calibration)), flush=True)
    return 0


async def _calibration(
    composition: CompositionRoot, part: ModelServerPart, ovms_models: Path | None
) -> ComponentCalibration:
    if ovms_models is None:
        return await composition.calibration(part)
    return await composition.ovms_calibration(part, ovms_models)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``rag-calibrate``; ``RAG_*`` overrides are honoured, as everywhere.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status: 0 when every requested component was calibrated, else 1.

    Example:
        >>> main(["--only", "reranker"])  # prints the reranker's .env lines
        0
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Calibrate the in-process embedder and reranker on this "
        "machine and print the RAG_* settings that pin the result, as .env lines.",
    )
    parser.add_argument("--only", choices=PARTS, help="calibrate just this model")
    parser.add_argument(
        "--ovms-models",
        type=Path,
        metavar="DIR",
        help="calibrate OpenVINO Model Server on the export in DIR (needs ovms on "
        "PATH) instead of the in-process models; it prints RAG_OVMS_NUM_STREAMS "
        "and RAG_OVMS_THREADS, pinned per App (one OVMS App serves one model)",
    )
    args = parser.parse_args(argv)
    # The Apps' log format, on stderr: stdout holds only the .env lines.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    parts = [args.only] if args.only else list(PARTS)
    return asyncio.run(calibrate_parts(parts, args.ovms_models))
