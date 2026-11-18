"""``qp-export-openvino``: convert the in-process models to OpenVINO, once.

The ``openvino`` runtime (``QP_LOCAL_MODELS__RUNTIME``, the default) loads the
folder this command writes for each model under ``QP_LOCAL_MODELS__OPENVINO_DIR``. An
App whose folder is missing exports it itself on its first start, so this command
only pre-warms: an image build, or a disk that should start without the Hub. The
models are the source ids of the workflow file's embedders and rerankers,
exported at the weight format of its ``local_models`` block
(``QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT``; an ``int8`` export goes to
``<model>-int8``), so ``QP_*`` overrides apply.

Example:
    >>> qp-export-openvino --only reranker --models-dir models/openvino
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ..config.workflow_settings import WorkflowSettings
from ..domain.runtime.backends import OpenVinoWeightFormat
from ..infrastructure.local_models.local_model import exported_model_dir
from ..infrastructure.local_models.openvino_export import (
    OpenVinoUnavailableError,
    export_embedder,
    export_reranker,
)

PROG = "qp-export-openvino"
# Failures an operator can act on (missing package, download or disk problem, a
# model the converter refuses); anything else is a bug and keeps its traceback.
EXPECTED_ERRORS = (OpenVinoUnavailableError, OSError, ValueError)


def export_part(
    component_type: str,
    model_id: str,
    target: Path,
    weight_format: OpenVinoWeightFormat,
) -> Path:
    """Export one model with the converter its component type needs.

    Args:
        component_type: ``embedder`` or ``reranker``.
        model_id: The model's Hugging Face id or local path.
        target: The folder to write.
        weight_format: ``fp32`` or ``int8``, what the weights are stored as.

    Returns:
        ``target``, now holding the export.

    Example:
        >>> export_part("reranker", "org/rerank", Path("models/openvino/rerank"),
        ...             "fp32")
        PosixPath('models/openvino/rerank')
    """
    exporters: dict[str, Callable[[str, Path, OpenVinoWeightFormat], Path]] = {
        "embedder": export_embedder,
        "reranker": export_reranker,
    }
    return exporters[component_type](model_id, target, weight_format)


def exported_components(settings: WorkflowSettings) -> list[str]:
    """Return the components with an in-process model: embedders and rerankers.

    Args:
        settings: The settings whose workflow file declares the components.

    Returns:
        Their names, in the file's order.

    Example:
        >>> exported_components(WorkflowSettings())
        ['embedder', 'reranker']
    """
    types = ("embedder", "reranker")
    return [n for n, c in settings.components.items() if c.type in types]


def export_models(
    settings: WorkflowSettings, names: Sequence[str], models_dir: str
) -> int:
    """Export each component's model, one line per model; stop at the first failure.

    Args:
        settings: Whose components give the types and the source ids, and
            whose ``local_models`` block gives the weight format they are
            exported at.
        names: Which components' models to export, in order.
        models_dir: The folder that holds every export.

    Returns:
        0 once every model is exported; 1 after an expected failure, reported
        on stderr with the model and the folder.

    Example:
        >>> export_models(WorkflowSettings(), ["reranker"], "models/openvino")
        0
    """
    for name in names:
        if not _exported(settings, name, models_dir):
            return 1
    return 0


def _exported(settings: WorkflowSettings, name: str, models_dir: str) -> bool:
    component = settings.model_components()[name]
    model_id = component.source_id
    weight_format = settings.local_models.openvino_weight_format
    target = exported_model_dir(models_dir, model_id, weight_format=weight_format)
    try:
        export_part(component.type, model_id, target, weight_format)
    except EXPECTED_ERRORS as exc:
        print(
            f"{PROG}: cannot export {name} {model_id} to {target}: {exc}",
            file=sys.stderr,
        )
        return False
    print(f"{PROG}: {name} {model_id} ({weight_format}) -> {target}")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``qp-export-openvino``; ``QP_*`` overrides are honoured, as everywhere.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status: 0 when every requested model was exported, else 1; a
        name that is no embedder or reranker exits as a usage error.

    Example:
        >>> main(["--only", "embedder"])  # prints where the embedder went
        0
    """
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        settings = WorkflowSettings()
    except ValueError as exc:  # a settings mistake is a usage error, not a crash
        parser.error(str(exc))
    names = exported_components(settings)
    if args.only and args.only not in names:
        parser.error(
            f"--only {args.only!r}: expected an embedder or a reranker of the "
            f"workflow file: {', '.join(names)}"
        )
    models_dir = args.models_dir or settings.local_models.openvino_dir
    return export_models(settings, [args.only] if args.only else names, models_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Convert the embedders and rerankers to OpenVINO IR, one folder "
        "per model and weight format, where the openvino runtime loads them from.",
    )
    parser.add_argument(
        "--only", metavar="NAME", help="export just this component's model"
    )
    parser.add_argument(
        "--models-dir",
        help="where the exports go (default: QP_LOCAL_MODELS__OPENVINO_DIR)",
    )
    return parser
