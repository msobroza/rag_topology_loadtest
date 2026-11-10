"""``rag-served-models``: print the served models as shell assignments.

The ``RagSettings`` defaults are the one pin for the served models (a source id and
an alias each). ``ovms/export_models.sh`` evaluates this command's output instead
of repeating the four names::

    eval "$(rag-served-models)"
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Sequence

from ..config.settings import RagSettings


def served_model_assignments(settings: RagSettings) -> list[str]:
    """``NAME=value`` lines, shell-quoted so the output is safe to ``eval``.

    The names are the variables ``ovms/export_models.sh`` passes to the exporter.

    Example::

        served_model_assignments(RagSettings())[0]
        # 'RERANKER_SOURCE_MODEL=BAAI/bge-reranker-base'
    """
    values = {
        "RERANKER_SOURCE_MODEL": settings.reranker_model,
        "RERANKER_MODEL_NAME": settings.ovms_rerank_model,
        "EMBEDDER_SOURCE_MODEL": settings.embedder_model,
        "EMBEDDER_MODEL_NAME": settings.ovms_embeddings_model,
    }
    return [f"{name}={shlex.quote(value)}" for name, value in values.items()]


def main(argv: Sequence[str] | None = None) -> int:
    """``rag-served-models``; ``RAG_*`` overrides are honoured, as everywhere.

    Example::

        main([])  # prints RERANKER_SOURCE_MODEL=BAAI/bge-reranker-base, ...
    """
    parser = argparse.ArgumentParser(
        prog="rag-served-models",
        description="Print the served models (source id and alias) as shell "
        "assignments, for eval in ovms/export_models.sh.",
    )
    parser.parse_args(argv)
    print("\n".join(served_model_assignments(RagSettings())))
    return 0
