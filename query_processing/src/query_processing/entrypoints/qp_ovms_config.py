"""``qp-ovms-config``: the config an OVMS App starts with, calibrated first.

``domino/model_servers/ovms.sh`` runs it when an OVMS App starts. It refuses an
export of another weight format than the settings ask (the script sees the
variable only; the workflow file may state the format), calibrates
each of ``QP_OVMS_NUM_STREAMS`` and ``QP_OVMS_THREADS`` left at 0 (one ``ovms``
start per candidate, so ``ovms`` must be on ``PATH``), writes the serving folder
for the choice in ``--serving-dir`` (the export in ``--models-dir`` is never
written), and prints the config file's path. Standard output holds that path
and nothing else, for the script to read; the logs go to standard error. A
failure an operator can act on is one line on standard error and exit code 1; a
``--serve`` that names no embedder or reranker of the workflow file is a usage
error.

Example:
    $ qp-ovms-config --serve reranker --models-dir /mnt/data/ovms-models
          --serving-dir "$(mktemp -d)"
    /tmp/tmp.x1/config_reranker.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from ..composition.composition_root import CompositionRoot
from ..config.ovms_settings import OvmsSettings
from ..config.workflow_settings import WorkflowSettings
from ..domain.errors import RagDependencyError
from ..infrastructure.one_line_failure import one_line_failure
from ..infrastructure.ovms.ovms_process import OvmsStartError

PROG = "qp-ovms-config"
# Failures an operator can act on: a bad setting, an export or ovms that is
# missing, an export of another weight format, an ovms that would not start or
# answer, a graph it cannot rewrite.
EXPECTED_ERRORS = (OvmsStartError, RagDependencyError, OSError, ValueError)


class ServingFolders(NamedTuple):
    """The two folders of an OVMS App: the export it reads, the one it serves from.

    Example:
        >>> ServingFolders(Path("/mnt/data/ovms-models"), Path("/tmp/serving"))
    """

    models_dir: Path  # the folder OVMS's exporter wrote into; never written
    serving_dir: Path  # the App's own folder, written for the choice


# The component types OVMS's exporter exports, so the ones it can serve.
SERVED_TYPES = ("embedder", "reranker")


def served_components(settings: WorkflowSettings) -> list[str]:
    """Return the components OVMS can serve: the embedders and the rerankers.

    Args:
        settings: The settings whose workflow file declares the components.

    Returns:
        Their names, in the file's order.

    Example:
        >>> served_components(WorkflowSettings())
        ['embedder', 'reranker']
    """
    return [n for n, c in settings.components.items() if c.type in SERVED_TYPES]


async def write_serving_config(
    settings: WorkflowSettings, ovms: OvmsSettings, name: str, folders: ServingFolders
) -> int:
    """Calibrate OVMS for a component as the settings ask, then print its config path.

    Args:
        settings: The workflow settings the component is read from.
        ovms: The OVMS App's own settings: its pinned counts, its start timeout.
        name: An embedder or reranker component of the workflow file.
        folders: OVMS's export, and the App's own folder OVMS will serve from.

    Returns:
        0 once the path is printed; 1 after an expected failure, reported as one
        line on stderr.

    Example:
        >>> await write_serving_config(settings, ovms, "reranker", folders)
        0
    """
    async with CompositionRoot(settings) as composition:
        try:
            config = await composition.ovms_serving_config(
                name, ovms, folders.models_dir, folders.serving_dir
            )
        except EXPECTED_ERRORS as exc:
            return one_line_failure(PROG, f"cannot write the {name} config", exc)
    print(config, flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``qp-ovms-config``; ``QP_*`` overrides are honoured, as everywhere.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status: 0 when the config path was printed, else 1; a name
        that is no embedder or reranker exits as a usage error.

    Example:
        >>> main(["--serve", "reranker", "--models-dir", "m", "--serving-dir", "s"])
        0
    """
    parser = _parser()
    args = parser.parse_args(argv)
    # The Apps' log format, on stderr: stdout holds only the config path.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        settings, ovms = WorkflowSettings(), OvmsSettings()
    except ValueError as exc:  # pydantic's ValidationError is one
        return one_line_failure(PROG, "cannot read the settings", exc)
    names = served_components(settings)
    if args.serve not in names:
        parser.error(
            f"--serve {args.serve!r}: expected an embedder or a reranker of the "
            f"workflow file: {', '.join(names)}"
        )
    folders = ServingFolders(args.models_dir, args.serving_dir)
    return asyncio.run(write_serving_config(settings, ovms, args.serve, folders))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Calibrate OpenVINO Model Server's streams and threads left at "
        "0 by QP_OVMS_NUM_STREAMS and QP_OVMS_THREADS, write the folder it "
        "serves one component from, and print the config file to start it with.",
    )
    parser.add_argument(
        "--serve",
        metavar="NAME",
        required=True,
        help="the embedder or reranker component OVMS serves, such as reranker",
    )
    parser.add_argument(
        "--models-dir", type=Path, required=True, help="OVMS's export, read only"
    )
    parser.add_argument(
        "--serving-dir", type=Path, required=True, help="the App's own folder"
    )
    return parser
