"""``rag-served-models``: print the served models as shell assignments.

The ``RagSettings`` defaults are the one pin for the served models (a source id and
an alias each). ``ovms/export_models.sh`` evaluates this command's output instead
of repeating the four names.

Example:
    >>> eval "$(rag-served-models)"
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Sequence

from ..config.settings import RagSettings


def served_model_assignments(settings: RagSettings) -> list[str]:
    """Return ``NAME=value`` lines, shell-quoted so the output is safe to ``eval``.

    The names are the variables ``ovms/export_models.sh`` passes to the exporter.

    Args:
        settings: Source of the reranker and embedder source ids and served aliases.

    Returns:
        Four lines, in order: ``RERANKER_SOURCE_MODEL``, ``RERANKER_MODEL_NAME``,
        ``EMBEDDER_SOURCE_MODEL``, ``EMBEDDER_MODEL_NAME``.

    Example:
        >>> served_model_assignments(RagSettings())[0]
        'RERANKER_SOURCE_MODEL=antoinelouis/crossencoder-camembert-base-mmarcoFR'
    """
    values = {
        "RERANKER_SOURCE_MODEL": settings.rerank_model,
        "RERANKER_MODEL_NAME": settings.rerank_api_model,
        "EMBEDDER_SOURCE_MODEL": settings.embedding_model,
        "EMBEDDER_MODEL_NAME": settings.embedding_api_model,
    }
    return [f"{name}={shlex.quote(value)}" for name, value in values.items()]


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``rag-served-models``; ``RAG_*`` overrides are honoured, as everywhere.

    Args:
        argv: Command-line arguments (it takes none but ``--help``); ``None``
            reads ``sys.argv``.

    Returns:
        The exit status, 0, after printing the assignments one per line.

    Example:
        >>> main([])  # prints RERANKER_SOURCE_MODEL=<source id>, then three more
        0
    """
    parser = argparse.ArgumentParser(
        prog="rag-served-models",
        description="Print the served models (source id and alias) as shell "
        "assignments, for eval in ovms/export_models.sh.",
    )
    parser.parse_args(argv)
    print("\n".join(served_model_assignments(RagSettings())))
    return 0
