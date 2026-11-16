"""``rag-served-models``: print the served models as shell assignments.

The ``RagSettings`` defaults are the one pin for the served models (a source id and
an alias each), for the weight format they are exported at, and for the
generator OVMS pulls and its two batching options.
``ovms/export_models.sh`` evaluates this command's output instead of repeating the
names, and exports at the format it prints; ``domino/app.sh`` evaluates it to
start the ``ovms-generator`` Role.

Example:
    >>> eval "$(rag-served-models)"
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Sequence

from ..config.rag_settings import RagSettings


def served_model_assignments(settings: RagSettings) -> list[str]:
    """Return ``NAME=value`` lines, shell-quoted so the output is safe to ``eval``.

    The first five are the variables ``ovms/export_models.sh`` passes to the
    exporter; the last four are what ``domino/app.sh`` passes to ``ovms`` for the
    generator. A batched-token count of 0 prints empty, so the shell leaves the
    flag out and OVMS keeps its own default.

    Args:
        settings: Source of the reranker and embedder source ids and served
            aliases, of the OpenVINO weight format, and of the generator's
            source id, alias and batching options.

    Returns:
        Nine lines, in order: ``RERANKER_SOURCE_MODEL``, ``RERANKER_MODEL_NAME``,
        ``EMBEDDER_SOURCE_MODEL``, ``EMBEDDER_MODEL_NAME``,
        ``OPENVINO_WEIGHT_FORMAT``, ``GENERATOR_SOURCE_MODEL``,
        ``GENERATOR_MODEL_NAME``, ``GENERATOR_DYNAMIC_SPLIT_FUSE`` (``true`` or
        ``false``), ``GENERATOR_MAX_BATCHED_TOKENS`` (empty for 0).

    Example:
        >>> served_model_assignments(RagSettings())[0]
        'RERANKER_SOURCE_MODEL=antoinelouis/crossencoder-camembert-base-mmarcoFR'
    """
    values = {
        "RERANKER_SOURCE_MODEL": settings.rerank_model,
        "RERANKER_MODEL_NAME": settings.rerank_api_model,
        "EMBEDDER_SOURCE_MODEL": settings.embedding_model,
        "EMBEDDER_MODEL_NAME": settings.embedding_api_model,
        "OPENVINO_WEIGHT_FORMAT": settings.openvino_weight_format,
        "GENERATOR_SOURCE_MODEL": settings.ovms_generation_model,
        "GENERATOR_MODEL_NAME": settings.generation_api_model,
        "GENERATOR_DYNAMIC_SPLIT_FUSE": str(
            settings.ovms_generation_dynamic_split_fuse
        ).lower(),
        "GENERATOR_MAX_BATCHED_TOKENS": _batched_tokens(settings),
    }
    return [f"{name}={shlex.quote(value)}" for name, value in values.items()]


def _batched_tokens(settings: RagSettings) -> str:
    """OVMS's ``--max_num_batched_tokens`` as printed.

    Args:
        settings: Source of ``ovms_generation_max_batched_tokens``.

    Returns:
        The count, or ``""`` for 0, so ``domino/app.sh`` leaves the flag out and
        OVMS keeps its own default.
    """
    tokens = settings.ovms_generation_max_batched_tokens
    return str(tokens) if tokens else ""


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``rag-served-models``; ``RAG_*`` overrides are honoured, as everywhere.

    Args:
        argv: Command-line arguments (it takes none but ``--help``); ``None``
            reads ``sys.argv``.

    Returns:
        The exit status, 0, after printing the assignments one per line.

    Example:
        >>> main([])  # prints RERANKER_SOURCE_MODEL=<source id>, then eight more
        0
    """
    parser = argparse.ArgumentParser(
        prog="rag-served-models",
        description="Print the served models (source id and alias), the "
        "OpenVINO weight format and the generator's OVMS options as shell "
        "assignments, for eval in ovms/export_models.sh and domino/app.sh.",
    )
    parser.parse_args(argv)
    print("\n".join(served_model_assignments(RagSettings())))
    return 0
