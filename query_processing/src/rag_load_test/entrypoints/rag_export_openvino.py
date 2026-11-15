"""``rag-export-openvino``: convert the in-process models to OpenVINO, once.

The ``openvino`` runtime (``RAG_LOCAL_MODELS__RUNTIME``, the default) loads the
folder this command writes for each model under ``RAG_LOCAL_MODELS__OPENVINO_DIR``. An
App whose folder is missing exports it itself on its first start, so this command
only pre-warms: an image build, or a disk that should start without the Hub. The
models are the source ids of the workflow file's embedders and rerankers, so
``RAG_*`` overrides apply.

Example:
    >>> rag-export-openvino --only reranker --models-dir models/openvino
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ..config.rag_settings import RagSettings
from ..infrastructure.local_models.local_model import exported_model_dir
from ..infrastructure.local_models.openvino_export import (
    OpenVinoUnavailableError,
    export_embedder,
    export_reranker,
)

PROG = "rag-export-openvino"
# Failures an operator can act on (missing package, download or disk problem, a
# model the converter refuses); anything else is a bug and keeps its traceback.
EXPECTED_ERRORS = (OpenVinoUnavailableError, OSError, ValueError)


def export_part(component_type: str, model_id: str, target: Path) -> Path:
    """Export one model with the converter its component type needs.

    Args:
        component_type: ``embedder`` or ``reranker``.
        model_id: The model's Hugging Face id or local path.
        target: The folder to write.

    Returns:
        ``target``, now holding the export.

    Example:
        >>> export_part("reranker", "org/rerank", Path("models/openvino/rerank"))
        PosixPath('models/openvino/rerank')
    """
    exporters: dict[str, Callable[[str, Path], Path]] = {
        "embedder": export_embedder,
        "reranker": export_reranker,
    }
    return exporters[component_type](model_id, target)


def exported_components(settings: RagSettings) -> list[str]:
    """Return the components with an in-process model: embedders and rerankers.

    Args:
        settings: The settings whose workflow file declares the components.

    Returns:
        Their names, in the file's order.

    Example:
        >>> exported_components(RagSettings())
        ['embedder', 'reranker']
    """
    types = ("embedder", "reranker")
    return [n for n, c in settings.components.items() if c.type in types]


def export_models(settings: RagSettings, names: Sequence[str], models_dir: str) -> int:
    """Export each component's model, one line per model; stop at the first failure.

    Args:
        settings: Whose components give the types and the source ids.
        names: Which components' models to export, in order.
        models_dir: The folder that holds every export.

    Returns:
        0 once every model is exported; 1 after an expected failure, reported
        on stderr with the model and the folder.

    Example:
        >>> export_models(RagSettings(), ["reranker"], "models/openvino")
        0
    """
    for name in names:
        if not _exported(settings, name, models_dir):
            return 1
    return 0


def _exported(settings: RagSettings, name: str, models_dir: str) -> bool:
    component = settings.model_components()[name]
    model_id = component.source_id
    target = exported_model_dir(models_dir, model_id)
    try:
        export_part(component.type, model_id, target)
    except EXPECTED_ERRORS as exc:
        print(
            f"{PROG}: cannot export {name} {model_id} to {target}: {exc}",
            file=sys.stderr,
        )
        return False
    print(f"{PROG}: {name} {model_id} -> {target}")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``rag-export-openvino``; ``RAG_*`` overrides are honoured, as everywhere.

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
        settings = RagSettings()
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
        "per model, where the openvino runtime loads them from.",
    )
    parser.add_argument(
        "--only", metavar="NAME", help="export just this component's model"
    )
    parser.add_argument(
        "--models-dir",
        help="where the exports go (default: RAG_LOCAL_MODELS__OPENVINO_DIR)",
    )
    return parser
