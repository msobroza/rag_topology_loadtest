"""``rag-calibrate``: calibrate the CPU-bound stages here, print what pins them.

An App calibrates each in-process model parameter left at 0 when it starts, and
every start and every replica does it again. This command does it once, on a
machine of the App's hardware tier that holds the models, and prints what it
found as ``.env`` lines: one ``#`` comment per component, then the
environment paths of its ``local`` block that pin the choice, as
``KEY=VALUE`` lines. Pinned in ``deploy.yaml``'s ``settings`` or ``.env``,
they make an App start without calibrating, and every replica run the same
values.

It calibrates the component ``--only`` names, else every embedder and reranker
of the workflow file. Each model loads as an App would load it (``RAG_*``
overrides honoured), whatever the topology or Backend says, and every parameter
is searched, pinned or not.
The logs, ``stage_calibrated`` included, go to standard error, so standard
output can be appended to ``.env`` as it is.

Example:
    $ RAG_CALIBRATION_ROUNDS=3 rag-calibrate --only reranker >> .env
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

from ..composition.calibrated_stages import ComponentCalibration
from ..composition.composition_root import CompositionRoot
from ..composition.registered_names import check_registered_names
from ..config.rag_settings import RagSettings
from ..domain.calibration import CalibrationReport
from ..domain.errors import RagDependencyError
from ..infrastructure.local_models.model_provisioning import ModelProvisioningError
from ..infrastructure.local_models.openvino_export import OpenVinoUnavailableError

PROG = "rag-calibrate"
# The component types that run a model in-process, so can be calibrated.
CALIBRATED_TYPES = ("embedder", "reranker")
# Failures an operator can act on (a bad setting, a model that cannot be found,
# downloaded or exported, a missing package, a dependency that would not open);
# anything else is a bug and keeps its traceback.
EXPECTED_ERRORS = (
    ModelProvisioningError,
    OpenVinoUnavailableError,
    RagDependencyError,
    OSError,
    ValueError,
)


def env_lines(calibration: ComponentCalibration) -> list[str]:
    """Return a component's lines for a ``.env`` file: one comment, then its settings.

    Args:
        calibration: One component's calibration.

    Returns:
        The ``#`` comment, then one ``KEY=VALUE`` line per pinned setting.

    Example:
        >>> env_lines(calibration)[1:]
        ['RAG_COMPONENTS__RERANKER__LOCAL__THREADS=2',
         'RAG_COMPONENTS__RERANKER__LOCAL__BATCH_SIZE=2']
    """
    report = calibration.report
    where = f"timed on {calibration.cpus:g} CPUs under {calibration.runtime}"
    comment = (
        f"# {calibration.component} ({report.stage}): {where}, "
        f"{_choice(report)}; {_search(report)}"
    )
    return [comment, *(f"{name}={value}" for name, value in calibration.pinned.items())]


def _choice(report: CalibrationReport) -> str:
    seconds = report.chosen_seconds
    if seconds is None:  # one candidate per parameter: nothing to compare
        return "the only candidate, not timed"
    return f"{seconds:.3f} s per request"


def _search(report: CalibrationReport) -> str:
    tried = len(report.trials)
    noun = "candidate" if tried == 1 else "candidates"
    text = f"tried {tried} {noun} in {report.seconds:.1f} s"
    if report.cut_short:
        text += f", cut short by RAG_CALIBRATION_BUDGET_S={report.budget_s:g}"
    return text


def calibrated_components(settings: RagSettings) -> list[str]:
    """Return the components that run a model in-process: embedders and rerankers.

    Args:
        settings: The settings whose workflow file declares the components.

    Returns:
        Their names, in the file's order.

    Example:
        >>> calibrated_components(RagSettings())
        ['embedder', 'reranker']
    """
    return [n for n, c in settings.components.items() if c.type in CALIBRATED_TYPES]


async def calibrate_components(settings: RagSettings, names: Sequence[str]) -> int:
    """Calibrate each component in turn, printing its lines as soon as it is done.

    Args:
        settings: The settings the components are opened from.
        names: Which components to calibrate, in order.

    Returns:
        0 once every component is printed; 1 after an expected failure,
        reported as one line on stderr.

    Example:
        >>> await calibrate_components(RagSettings(), ["reranker"])  # its lines
        0
    """
    async with CompositionRoot(settings) as composition:
        return await _print_each(composition, names)


async def _print_each(composition: CompositionRoot, names: Sequence[str]) -> int:
    # Each component is printed as soon as it is done, so a later failure keeps it.
    for name in names:
        try:
            calibration = await composition.calibration(name)
        except EXPECTED_ERRORS as exc:
            return _failed(f"cannot calibrate {name}", exc)
        print("\n".join(env_lines(calibration)), flush=True)
    return 0


def _failed(what: str, exc: BaseException) -> int:
    # One line: pydantic's message spans several, and the value it names is on one.
    detail = " ".join(str(exc).split())
    print(f"{PROG}: {what}: {detail}", file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``rag-calibrate``; ``RAG_*`` overrides are honoured, as everywhere.

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
        settings = RagSettings()
        check_registered_names(settings)
    except ValueError as exc:  # pydantic's ValidationError is one
        return _failed("cannot read the settings", exc)
    names = calibrated_components(settings)
    if args.only and args.only not in names:
        parser.error(
            f"--only {args.only!r}: expected an embedder or a reranker of the "
            f"workflow file: {', '.join(names)}"
        )
    return asyncio.run(
        calibrate_components(settings, [args.only] if args.only else names)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Calibrate the in-process embedders and rerankers on this "
        "machine and print the environment paths that pin the result, as .env "
        "lines.",
    )
    parser.add_argument(
        "--only",
        metavar="NAME",
        help="calibrate just this component, such as reranker",
    )
    return parser
