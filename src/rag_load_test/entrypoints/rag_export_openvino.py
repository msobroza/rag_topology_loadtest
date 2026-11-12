"""``rag-export-openvino``: convert the in-process models to OpenVINO, once.

The ``openvino`` runtime (``RAG_LOCAL_MODEL_RUNTIME``, the default) loads the
folder this command writes for each model under ``RAG_OPENVINO_MODELS_DIR``. An
App whose folder is missing exports it itself on its first start, so this command
only pre-warms: an image build, or a disk that should start without the Hub. The
models are the settings' ``embedding_model`` and ``rerank_model``, so ``RAG_*``
overrides apply.

Example:
    >>> rag-export-openvino --only reranker --models-dir models/openvino
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ..adapters.local_models.local_model import exported_model_dir
from ..adapters.local_models.openvino_export import (
    OpenVinoUnavailableError,
    export_embedder,
    export_reranker,
)
from ..config.rag_settings import RagSettings

PROG = "rag-export-openvino"
ROLES = ("embedder", "reranker")
# Failures an operator can act on (missing package, download or disk problem, a
# model the converter refuses); anything else is a bug and keeps its traceback.
EXPECTED_ERRORS = (OpenVinoUnavailableError, OSError, ValueError)


def export_role(role: str, model_id: str, target: Path) -> Path:
    """Export one model with the converter its role needs.

    Args:
        role: ``embedder`` or ``reranker``.
        model_id: The model's Hugging Face id or local path.
        target: The folder to write.

    Returns:
        ``target``, now holding the export.

    Example:
        >>> export_role("reranker", "org/rerank", Path("models/openvino/rerank"))
        PosixPath('models/openvino/rerank')
    """
    exporters: dict[str, Callable[[str, Path], Path]] = {
        "embedder": export_embedder,
        "reranker": export_reranker,
    }
    return exporters[role](model_id, target)


def export_models(settings: RagSettings, roles: Sequence[str], models_dir: str) -> int:
    """Export each role's model, one line per model; stop at the first failure.

    Args:
        settings: Source of the embedder and reranker model ids.
        roles: Which models to export, in order.
        models_dir: The folder that holds every export.

    Returns:
        0 once every model is exported; 1 after an expected failure, reported
        on stderr with the model and the folder.

    Example:
        >>> export_models(RagSettings(), ["reranker"], "models/openvino")
        0
    """
    model_ids = {
        "embedder": settings.embedding_model,
        "reranker": settings.rerank_model,
    }
    for role in roles:
        target = exported_model_dir(models_dir, model_ids[role])
        try:
            export_role(role, model_ids[role], target)
        except EXPECTED_ERRORS as exc:
            print(
                f"{PROG}: cannot export {role} {model_ids[role]} to {target}: {exc}",
                file=sys.stderr,
            )
            return 1
        print(f"{PROG}: {role} {model_ids[role]} -> {target}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``rag-export-openvino``; ``RAG_*`` overrides are honoured, as everywhere.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status: 0 when every requested model was exported, else 1.

    Example:
        >>> main(["--only", "embedder"])  # prints where the embedder went
        0
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Convert the embedder and reranker to OpenVINO IR, one folder "
        "per model, where the openvino runtime loads them from.",
    )
    parser.add_argument("--only", choices=ROLES, help="export just this model")
    parser.add_argument(
        "--models-dir", help="where the exports go (default: RAG_OPENVINO_MODELS_DIR)"
    )
    args = parser.parse_args(argv)
    settings = RagSettings()
    roles = [args.only] if args.only else list(ROLES)
    return export_models(
        settings, roles, args.models_dir or settings.openvino_models_dir
    )
