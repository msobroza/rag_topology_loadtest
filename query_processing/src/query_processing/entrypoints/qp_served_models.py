"""``qp-served-models``: print the served models, one line of shell words each.

The workflow file is the one pin for the served models: each embedder and
reranker it declares, by any name, with a source id and an alias, is a model an
OVMS model-server App may serve. Its ``local_models`` block is the one pin for
the weight format they are exported at, the same for all, which the in-process
export reads too. ``ovms/export_models.sh`` reads this command's output to
export each one under its component's name and at the format printed, instead
of repeating the names.

Example:
    >>> qp-served-models
    embedder embedder Snowflake/snowflake-arctic-embed-l-v2.0 snowflake-arc... fp32
    reranker reranker antoinelouis/crossencoder-camembert-base-mmarcoFR cross... fp32
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Sequence

from ..config.workflow_settings import WorkflowSettings


def served_model_lines(settings: WorkflowSettings) -> list[str]:
    """Return one line per embedder and reranker, ending with the weight format.

    A line holds the component's name, type, source id and alias, then the
    OpenVINO weight format, the same on every line: it is one setting. Each
    word is shell-quoted, so a line is safe to ``eval "set -- $line"``, as
    ``ovms/export_models.sh`` does.

    Args:
        settings: Whose workflow file declares the components and the weight
            format (its ``local_models`` block).

    Returns:
        The lines, in the file's order; an LLM or a vector store has none.

    Example:
        >>> served_model_lines(WorkflowSettings())[1]
        'reranker reranker antoinelouis/crossencoder-camembert-base-mmarcoFR ... fp32'
    """
    exported = ("embedder", "reranker")  # the types the OVMS exporter exports
    weight_format = settings.local_models.openvino_weight_format
    return [
        shlex.join(
            [name, component.type, component.source_id, component.alias, weight_format]
        )
        for name, component in settings.model_components().items()
        if component.type in exported
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``qp-served-models``; ``QP_*`` overrides are honoured, as everywhere.

    Args:
        argv: Command-line arguments (it takes none but ``--help``); ``None``
            reads ``sys.argv``.

    Returns:
        The exit status, 0, after printing one line per served model.

    Example:
        >>> main([])  # prints "embedder embedder <source id> <alias> fp32", then more
        0
    """
    parser = argparse.ArgumentParser(
        prog="qp-served-models",
        description="Print each embedder and reranker of the workflow file as one "
        "line of shell words (name, type, source id, alias, OpenVINO weight "
        "format), for ovms/export_models.sh.",
    )
    parser.parse_args(argv)
    try:
        settings = WorkflowSettings()
    except ValueError as exc:  # a settings mistake is a usage error, not a crash
        parser.error(str(exc))
    print("\n".join(served_model_lines(settings)))
    return 0
