"""``rag-ovms-config``: the config an OVMS App starts with, calibrated first.

``domino/app.sh`` runs it when an OVMS App starts. It calibrates each of
``RAG_OVMS_NUM_STREAMS`` and ``RAG_OVMS_THREADS`` left at 0 (one ``ovms`` start
per candidate, so ``ovms`` must be on ``PATH``), writes the serving folder for
the choice in ``--serving-dir`` (the export in ``--models-dir`` is never
written), and prints the config file's path. Standard output holds that path
and nothing else, for the script to read; the logs go to standard error. A
failure an operator can act on is one line on standard error and exit code 1.

Example:
    $ rag-ovms-config --serve reranker --models-dir /mnt/data/ovms-models
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
from typing import get_args

from ..composition.composition_root import CompositionRoot
from ..config.rag_settings import RagSettings
from ..domain.errors import RagDependencyError
from ..domain.roles import ModelServerPart
from ..infrastructure.one_line_failure import one_line_failure
from ..infrastructure.ovms.ovms_process import OvmsStartError

PROG = "rag-ovms-config"
PARTS: tuple[ModelServerPart, ...] = get_args(ModelServerPart)  # embedder, reranker
# Failures an operator can act on: a bad setting, an export or ovms that is
# missing, an ovms that would not start or answer, a graph it cannot rewrite.
EXPECTED_ERRORS = (OvmsStartError, RagDependencyError, OSError, ValueError)


async def write_serving_config(
    part: ModelServerPart, models_dir: Path, serving_dir: Path
) -> int:
    """Calibrate OVMS for ``part`` as the settings ask, then print its config path.

    Args:
        part: ``embedder`` or ``reranker``.
        models_dir: The folder OVMS's exporter wrote into.
        serving_dir: The App's own folder OVMS will serve from.

    Returns:
        0 once the path is printed; 1 after an expected failure, reported as one
        line on stderr.

    Example:
        >>> await write_serving_config("reranker", models, serving)
        0
    """
    try:
        settings = RagSettings()
    except ValueError as exc:  # pydantic's ValidationError is one
        return one_line_failure(PROG, "cannot read the settings", exc)
    async with CompositionRoot(settings) as composition:
        try:
            config = await composition.ovms_serving_config(
                part, models_dir, serving_dir
            )
        except EXPECTED_ERRORS as exc:
            return one_line_failure(PROG, f"cannot write the {part} config", exc)
    print(config, flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``rag-ovms-config``; ``RAG_*`` overrides are honoured, as everywhere.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status: 0 when the config path was printed, else 1.

    Example:
        >>> main(["--serve", "reranker", "--models-dir", "m", "--serving-dir", "s"])
        0
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Calibrate OpenVINO Model Server's streams and threads left at "
        "0 by RAG_OVMS_NUM_STREAMS and RAG_OVMS_THREADS, write the folder it "
        "serves one model from, and print the config file to start it with.",
    )
    parser.add_argument("--serve", choices=PARTS, required=True, help="the model")
    parser.add_argument(
        "--models-dir", type=Path, required=True, help="OVMS's export, read only"
    )
    parser.add_argument(
        "--serving-dir", type=Path, required=True, help="the App's own folder"
    )
    args = parser.parse_args(argv)
    # The Apps' log format, on stderr: stdout holds only the config path.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    return asyncio.run(
        write_serving_config(args.serve, args.models_dir, args.serving_dir)
    )
