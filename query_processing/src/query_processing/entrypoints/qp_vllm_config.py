"""``qp-vllm-config``: what a vLLM App passes to ``vllm serve``, and exports first.

``domino/model_servers/vllm.sh`` runs it when a vLLM App starts. ``--serve NAME``
prints one line of shell words: the component's source id, then
``--served-model-name`` and its alias, both from the workflow file, then the
flags of ``QP_VLLM_*`` (``VllmSettings.serve_flags``). ``--environment NAME``
prints one line of ``NAME=value`` words, the CPU variables vLLM reads, for the
script to export. Standard output holds that line and nothing else. A failure
an operator can act on (a setting given twice, an empty source id) is one line
on standard error and exit code 1; a name that is no model component of the
workflow file is a usage error. Which component types a vLLM App may serve is
the deploy tool's to refuse, before an App starts.

Example:
    $ qp-vllm-config --serve reranker
    org/reranker --served-model-name reranker --kv-cache-memory-bytes 8589934592
    $ QP_VLLM_CPU_RESERVED_CPUS=2 qp-vllm-config --environment reranker
    VLLM_CPU_NUM_OF_RESERVED_CPU=2
"""

from __future__ import annotations

import argparse
import os
import shlex
from collections.abc import Mapping, Sequence

from ..config.apps.vllm_settings import VllmSettings
from ..config.workflow_settings import WorkflowSettings
from ..infrastructure.one_line_failure import one_line_failure

PROG = "qp-vllm-config"


def serve_words(settings: WorkflowSettings, vllm: VllmSettings, name: str) -> str:
    """Return the arguments of ``vllm serve`` for a component, as one line.

    Args:
        settings: The workflow settings the component is read from.
        vllm: The vLLM App's own settings.
        name: A model component of the workflow file.

    Returns:
        The source id, ``--served-model-name``, the alias, then
        ``vllm.serve_flags()``, shell-quoted.

    Raises:
        ValueError: If the component has no source id for vLLM to load.

    Example:
        >>> settings = WorkflowSettings(components={"llm": {"source_id": "org/m"}})
        >>> serve_words(settings, VllmSettings(kv_cache_gib=0), "llm")
        'org/m --served-model-name m'
    """
    component = settings.model_components()[name]
    if not component.source_id:
        variable = settings.variable_of("components", name, "source_id")
        raise ValueError(
            f"components.{name}.source_id is empty ({variable}), and vLLM loads the "
            "weights it names; set it in the workflow file, or in the deploy "
            "file's overrides, to a Hugging Face id or a folder of weights"
        )
    served = [component.source_id, "--served-model-name", component.alias]
    return shlex.join(served + vllm.serve_flags())


def environment_words(vllm: VllmSettings, environ: Mapping[str, str]) -> str:
    """Return the CPU variables a vLLM App exports, as one line of words.

    Args:
        vllm: The vLLM App's own settings.
        environ: The App's environment, checked against them.

    Returns:
        ``NAME=value`` words, shell-quoted; ``""`` when none is set.

    Raises:
        ValueError: If ``environ`` holds one at another value, or the legacy
            KV cache variable beside the cache's flag
            (``VllmSettings.exported_environment``).

    Example:
        >>> environment_words(VllmSettings(cpu_reserved_cpus=2), {})
        'VLLM_CPU_NUM_OF_RESERVED_CPU=2'
    """
    exported = vllm.exported_environment(environ)
    return shlex.join(f"{name}={value}" for name, value in exported.items())


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``qp-vllm-config``; ``QP_*`` overrides are honoured, as everywhere.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status: 0 when the line was printed, else 1; a name that is
        no model component exits as a usage error.

    Example:
        >>> main(["--serve", "reranker"])
        0
    """
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        settings, vllm = WorkflowSettings(), VllmSettings()
    except ValueError as exc:  # pydantic's ValidationError is one
        return one_line_failure(PROG, "cannot read the settings", exc)
    name = args.serve if args.serve is not None else args.environment
    names = list(settings.model_components())
    if name not in names:
        option = "--serve" if args.serve is not None else "--environment"
        parser.error(
            f"{option} {name!r}: expected a model component of the workflow file: "
            f"{', '.join(names)}"
        )
    try:
        # The environment is checked on --serve too: it is what vLLM starts on.
        line = environment_words(vllm, os.environ)
        if args.serve is not None:
            line = serve_words(settings, vllm, name)
    except ValueError as exc:
        return one_line_failure(PROG, f"cannot serve the {name}", exc)
    print(line, flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Print what a vLLM App passes to vllm serve (source id, "
        "served name, then the QP_VLLM_* flags), or the CPU variables it "
        "exports, as one line of shell words.",
    )
    asked = parser.add_mutually_exclusive_group(required=True)
    asked.add_argument(
        "--serve",
        metavar="NAME",
        help="the component the App serves: print vllm serve's arguments",
    )
    asked.add_argument(
        "--environment",
        metavar="NAME",
        help="the component the App serves: print the NAME=value words to export",
    )
    return parser
