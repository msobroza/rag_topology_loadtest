"""The embedder a vector store's searches use, read from the workflow graph.

A ``retrieve`` stage in the ``dense`` or ``hybrid`` retrieval mode reads its
query embedding first. The stage that writes that field calls the embedder
whose vectors the store is searched with, so a fake store is seeded with the
same one. With several embedder components in a workflow file, this is how the
one the searches use is told from the others (spec 2.5).

Example:
    >>> searched_embedder(settings.graph, settings.component_types())
    'embedder'
"""

from __future__ import annotations

from collections.abc import Mapping

from ..domain.graph_description import GraphDescription

# The stage type that searches the store, and the retrieval modes in which it
# reads an embedding, as the first of its reads.
_RETRIEVE_STAGE_TYPE = "retrieve"
_MODES_THAT_READ_AN_EMBEDDING = ("dense", "hybrid")


def searched_embedder(
    graph: GraphDescription, component_types: Mapping[str, str]
) -> str | None:
    """Return the embedder called by the stage that writes what retrieval reads.

    Args:
        graph: The workflow file's graph.
        component_types: Each component's type, by name.

    Returns:
        The embedder component's name; None when no stage writes the embedding
        a ``retrieve`` stage reads, or when several call different embedders.

    Example:
        >>> searched_embedder(graph, {"embedder": "embedder"})
        'embedder'
    """
    searched = _searched_embeddings(graph)
    embedders = {
        stage.call
        for stage in graph.stages.values()
        if component_types.get(stage.call) == "embedder"
        and searched.intersection(stage.writes)
    }
    return embedders.pop() if len(embedders) == 1 else None


def _searched_embeddings(graph: GraphDescription) -> set[str]:
    # The field each retrieve stage that searches by vector reads first: the
    # retrieve stage takes its embedding first in both modes (its
    # READS_BY_RETRIEVAL_MODE), which config cannot import from application.
    return {
        stage.reads[0]
        for name, stage in graph.stages.items()
        if stage.type_name(name) == _RETRIEVE_STAGE_TYPE
        and stage.stage_settings.get("retrieval_mode") in _MODES_THAT_READ_AN_EMBEDDING
        and stage.reads
    }
