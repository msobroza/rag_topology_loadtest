"""``qp-ovms-config``: the config an OVMS App starts with, calibrated first.

``domino/model_servers/ovms.sh`` runs it when an OVMS App starts. It refuses an
export of another weight format than the settings ask (the script sees the
variable only; the workflow file may state the format), calibrates
each of ``QP_OVMS_NUM_STREAMS`` and ``QP_OVMS_THREADS`` left at 0 (one ``ovms``
start per candidate, so ``ovms`` must be on ``PATH``), writes the serving folder
for the choice in ``--serving-dir`` (the export in ``--models-dir`` is never
written), and prints the config file's path. Standard output holds that path
and nothing else, for the script to read; the logs go to standard error. A
failure an operator can act on is one line on standard error and exit code 1,
such as a ``--serve`` of an LLM, whose config left in the models folder it names;
a ``--serve`` that names no embedder, reranker or LLM of the workflow file is a
usage error.

An LLM is not exported nor calibrated: OVMS pulls it from the Hub itself. For
its App, ``--serve-llm`` checks the settings (a source id to pull, no stream or
thread count pinned, nothing of the graph OVMS writes itself: the OpenVINO
block, the pool size, the embeddings options) and prints the flags of the LLM's
options, then the server's, as shell words on one line, empty at OVMS's own
values. An embedder's or a reranker's App refuses an LLM's option, and a
reranker's the embeddings options. ``--server-flags`` prints the server's flags
alone (``QP_OVMS_REST_WORKERS``, ..., then ``QP_OVMS_ARGS``), for an App of any
type, to append to the ``ovms`` it starts on the config.

Example:
    $ qp-ovms-config --serve-llm llm
    --dynamic_split_fuse false --max_num_batched_tokens 8192
    $ QP_OVMS_DYNAMIC_SPLIT_FUSE=true qp-ovms-config --serve-llm llm
    --max_num_batched_tokens 8192
    $ QP_OVMS_REST_WORKERS=4 qp-ovms-config --server-flags reranker
    --rest_workers 4
    $ qp-ovms-config --serve reranker --models-dir /mnt/data/ovms-models
          --serving-dir "$(mktemp -d)"
    /tmp/tmp.x1/config_reranker.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from ..composition.composition_root import CompositionRoot
from ..config.apps.ovms_settings import OvmsSettings
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


# The component types OVMS's exporter exports, so the ones served on a config
# of the export; OVMS serves an LLM too, pulled by itself (llm_components).
SERVED_TYPES = ("embedder", "reranker")
# The type whose graph takes QP_OVMS_EMBEDDINGS_*; an App of another refuses them.
EMBEDDINGS_TYPE = "embedder"


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


def llm_components(settings: WorkflowSettings) -> list[str]:
    """Return the LLMs of the workflow file, which OVMS pulls and serves itself.

    Args:
        settings: The settings whose workflow file declares the components.

    Returns:
        Their names, in the file's order.

    Example:
        >>> llm_components(WorkflowSettings())
        ['llm']
    """
    return [n for n, c in settings.components.items() if c.type == "llm"]


def llm_ovms_flags(settings: WorkflowSettings, ovms: OvmsSettings, name: str) -> str:
    """Check an LLM's App, then return its batching flags as one line of shell words.

    Args:
        settings: The workflow settings the LLM is read from.
        ovms: The OVMS App's own settings.
        name: An LLM component of the workflow file.

    Returns:
        The flags of ``ovms.llm_flags()``, then those of ``ovms.server_flags()``,
        shell-quoted; "" at OVMS's own values.

    Raises:
        ValueError: If the LLM has no source id for OVMS to pull, or a stream
            or thread count, the OpenVINO block, the graph's pool size or an
            embeddings option is set, which an LLM's App would ignore.

    Example:
        >>> settings = WorkflowSettings(components={"llm": {"source_id": "org/m-ov"}})
        >>> llm_ovms_flags(settings, OvmsSettings(max_batched_tokens=512), "llm")
        '--dynamic_split_fuse false --max_num_batched_tokens 512'
    """
    if not settings.model_components()[name].source_id:
        variable = settings.variable_of("components", name, "source_id")
        raise ValueError(
            f"components.{name}.source_id is empty ({variable}), and OVMS pulls the "
            "model it names from the Hugging Face Hub; set it in the workflow file, "
            "or in the deploy file's overrides, to an OpenVINO model such as "
            "OpenVINO/<model>-int4-ov"
        )
    ovms.require_no_stream_pin(name)
    ovms.require_no_graph_option(name)
    return shlex.join(ovms.llm_flags() + ovms.server_flags())


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
        The exit status: 0 when the config path (or an LLM's flags) was
        printed, else 1; a name that is no component of the asked type exits
        as a usage error.

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
    if args.server_flags is not None:
        return _print_server_flags(parser, settings, ovms, args.server_flags)
    if args.serve_llm is not None:
        return _serve_llm(parser, settings, ovms, args.serve_llm)
    return _serve_exported(parser, settings, ovms, args)


def _print_server_flags(
    parser: argparse.ArgumentParser,
    settings: WorkflowSettings,
    ovms: OvmsSettings,
    name: str,
) -> int:
    # Every App's: an embedder's, a reranker's, an LLM's. The settings were
    # checked when read, so only the name can be wrong.
    names = served_components(settings) + llm_components(settings)
    if name not in names:
        parser.error(
            f"--server-flags {name!r}: expected an embedder, a reranker or an LLM "
            f"of the workflow file: {', '.join(names)}"
        )
    print(shlex.join(ovms.server_flags()), flush=True)
    return 0


def _serve_llm(
    parser: argparse.ArgumentParser,
    settings: WorkflowSettings,
    ovms: OvmsSettings,
    name: str,
) -> int:
    names = llm_components(settings)
    if name not in names:
        parser.error(
            f"--serve-llm {name!r}: expected an LLM of the workflow file: "
            f"{', '.join(names)}"
        )
    try:
        flags = llm_ovms_flags(settings, ovms, name)
    except ValueError as exc:
        return one_line_failure(PROG, f"cannot serve the {name}", exc)
    print(flags, flush=True)
    return 0


def _serve_exported(
    parser: argparse.ArgumentParser,
    settings: WorkflowSettings,
    ovms: OvmsSettings,
    args: argparse.Namespace,
) -> int:
    names = served_components(settings)
    if args.serve in llm_components(settings):
        return _refuse_an_llms_config(args.serve, args.models_dir)
    if args.serve not in names:
        parser.error(
            f"--serve {args.serve!r}: expected an embedder or a reranker of the "
            f"workflow file: {', '.join(names)}"
        )
    missing = [
        flag
        for flag, value in (
            ("--models-dir", args.models_dir),
            ("--serving-dir", args.serving_dir),
        )
        if value is None
    ]
    if missing:
        parser.error(f"--serve {args.serve} needs {' and '.join(missing)}")
    try:
        _require_options_of_its_type(settings, ovms, args.serve)
    except ValueError as exc:
        return one_line_failure(PROG, f"cannot serve the {args.serve}", exc)
    folders = ServingFolders(args.models_dir, args.serving_dir)
    return asyncio.run(write_serving_config(settings, ovms, args.serve, folders))


def _require_options_of_its_type(
    settings: WorkflowSettings, ovms: OvmsSettings, name: str
) -> None:
    # By the type the workflow file gives the component, never by its name.
    ovms.require_no_llm_option(name)
    if settings.components[name].type != EMBEDDINGS_TYPE:
        ovms.require_no_embeddings_option(name)


def _refuse_an_llms_config(name: str, models_dir: Path | None) -> int:
    # ovms.sh asks --serve only when the folder holds config_<name>.json: an old
    # export, or a folder shared with another App, which the LLM never needs.
    config = (models_dir or Path()) / f"config_{name}.json"
    reason = ValueError(
        f"{config} is in its models folder, so its App took it for an export to "
        f"serve, but {name} is an LLM, which OVMS pulls from the Hugging Face Hub: "
        "an LLM is served without one; remove that file, or point "
        "QP_OVMS_MODELS_DIR at a folder of the LLM's own"
    )
    return one_line_failure(PROG, f"cannot serve the {name}", reason)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Calibrate OpenVINO Model Server's streams and threads left at "
        "0 by QP_OVMS_NUM_STREAMS and QP_OVMS_THREADS, write the folder it "
        "serves one component from, and print the config file to start it with; "
        "or, for an LLM, print the flags of its options; or print the ovms "
        "server's flags.",
    )
    _add_what_is_asked(parser)
    parser.add_argument(
        "--models-dir", type=Path, help="with --serve: OVMS's export, read only"
    )
    parser.add_argument(
        "--serving-dir", type=Path, help="with --serve: the App's own folder"
    )
    return parser


def _add_what_is_asked(parser: argparse.ArgumentParser) -> None:
    # One of the three, each naming the component of the App.
    served = parser.add_mutually_exclusive_group(required=True)
    served.add_argument(
        "--serve",
        metavar="NAME",
        help="the embedder or reranker component OVMS serves, such as reranker",
    )
    served.add_argument(
        "--serve-llm",
        metavar="NAME",
        help="the LLM component OVMS pulls and serves: print its flags, then the "
        "server's",
    )
    served.add_argument(
        "--server-flags",
        metavar="NAME",
        help="the component an App serves: print the ovms server's flags",
    )
