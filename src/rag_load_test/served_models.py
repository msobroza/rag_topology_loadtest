"""The served models, read from the ``RagSettings`` defaults (their one pin).

Each served model pairs a source id (the Hugging Face weights) with an alias (the
name clients send to a model server). ``ovms/export_models.sh`` and
``domino/Dockerfile`` ask this module instead of repeating the four names.

Example::

    embedder, reranker = served_models_from_settings(RagSettings())

``rag-served-models`` prints them as shell assignments for ``export_models.sh``::

    eval "$(rag-served-models)"
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from .settings import RagSettings


@dataclass(frozen=True)
class ServedModel:
    """A model as a model server exposes it: source id plus alias."""

    source_id: str
    alias: str


class ServedModelPair(NamedTuple):
    """The embedder's and the reranker's served model, in that order."""

    embedder: ServedModel
    reranker: ServedModel


def served_models_from_settings(settings: RagSettings) -> ServedModelPair:
    """The embedder's and the reranker's served model, as ``settings`` pins them.

    Example::

        served_models_from_settings(RagSettings()).reranker.alias
        # 'bge-reranker-base'
    """
    return ServedModelPair(
        embedder=ServedModel(settings.embedder_model, settings.ovms_embeddings_model),
        reranker=ServedModel(settings.reranker_model, settings.ovms_rerank_model),
    )


def served_models_shell_lines(models: ServedModelPair) -> list[str]:
    """``NAME=value`` lines, values shell-quoted so the output is safe to ``eval``.

    The names are the variables ``ovms/export_models.sh`` passes to the exporter.

    Example::

        served_models_shell_lines(served_models_from_settings(RagSettings()))[0]
        # 'RERANKER_SOURCE_MODEL=BAAI/bge-reranker-base'
    """
    assignments = [
        ("RERANKER_SOURCE_MODEL", models.reranker.source_id),
        ("RERANKER_MODEL_NAME", models.reranker.alias),
        ("EMBEDDER_SOURCE_MODEL", models.embedder.source_id),
        ("EMBEDDER_MODEL_NAME", models.embedder.alias),
    ]
    return [f"{name}={shlex.quote(value)}" for name, value in assignments]


def main(argv: Sequence[str] | None = None) -> int:
    """``rag-served-models``: print the served models as ``eval``-able assignments.

    ``RAG_*`` overrides (environment or ``.env``) are honoured, as everywhere.

    Example::

        main([])  # prints RERANKER_SOURCE_MODEL=BAAI/bge-reranker-base, ...
    """
    argparse.ArgumentParser(
        prog="rag-served-models",
        description="Print the served models (source id and alias) as shell "
        "assignments, for eval in ovms/export_models.sh.",
    ).parse_args(argv)
    for line in served_models_shell_lines(served_models_from_settings(RagSettings())):
        print(line)
    return 0
