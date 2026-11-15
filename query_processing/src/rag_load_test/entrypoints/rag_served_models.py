"""``rag-served-models``: print the served models, one line of shell words each.

The workflow file is the one pin for the served models: each embedder and
reranker it declares, by any name, with a source id and an alias, is a model an
OVMS model-server App may serve. ``ovms/export_models.sh`` reads this command's
output to export each one under its component's name, instead of repeating the
names.

Example:
    >>> rag-served-models
    embedder embedder Snowflake/snowflake-arctic-embed-l-v2.0 snowflake-arc...
    reranker reranker antoinelouis/crossencoder-camembert-base-mmarcoFR cross...
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Sequence

from ..config.rag_settings import RagSettings


def served_model_lines(settings: RagSettings) -> list[str]:
    """Return one line per embedder and reranker: its name, type, source id, alias.

    Each word is shell-quoted, so a line is safe to ``eval "set -- $line"``, as
    ``ovms/export_models.sh`` does.

    Args:
        settings: Whose workflow file declares the components.

    Returns:
        The lines, in the file's order; an LLM or a vector store has none.

    Example:
        >>> served_model_lines(RagSettings())[1]
        'reranker reranker antoinelouis/crossencoder-camembert-base-mmarcoFR ...'
    """
    exported = ("embedder", "reranker")  # the types the OVMS exporter exports
    return [
        shlex.join([name, component.type, component.source_id, component.alias])
        for name, component in settings.model_components().items()
        if component.type in exported
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``rag-served-models``; ``RAG_*`` overrides are honoured, as everywhere.

    Args:
        argv: Command-line arguments (it takes none but ``--help``); ``None``
            reads ``sys.argv``.

    Returns:
        The exit status, 0, after printing one line per served model.

    Example:
        >>> main([])  # prints "embedder embedder <source id> <alias>", then more
        0
    """
    parser = argparse.ArgumentParser(
        prog="rag-served-models",
        description="Print each embedder and reranker of the workflow file as one "
        "line of shell words (name, type, source id, alias), for "
        "ovms/export_models.sh.",
    )
    parser.parse_args(argv)
    print("\n".join(served_model_lines(RagSettings())))
    return 0
