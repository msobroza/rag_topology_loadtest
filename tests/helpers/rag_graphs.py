"""Today's RAG graph with some of its stages' settings changed, for the tests.

The graph is the one of the workflow file that ships with the code, so a test
names only what it changes. A lexical graph has no embed stage, as a lexical
workflow file has none: its first chain starts at retrieve. A test hands the
graph to the settings as ``graph=``, which replaces the file's whole graph.

Example::

    settings_factory(graph=rag_graph("hybrid", retrieve_top_k=3, rerank_top_k=3))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rag_load_test.config.workflow_file import PACKAGED_WORKFLOW
from rag_load_test.domain.graph_description import GraphDescription

# Each keyword of rag_graph: the stage it sets and the key of that stage.
STAGE_SETTING = {
    "retrieve_top_k": ("retrieve", "top_k"),
    "rrf_k": ("retrieve", "rrf_k"),
    "rerank_top_k": ("rerank", "top_k"),
    "system_prompt": ("generate", "system_prompt"),
    "user_prompt": ("generate", "user_prompt"),
}


def packaged_graph() -> dict[str, Any]:
    """The ``graph`` block of the workflow file that ships with the code, as read."""
    return yaml.safe_load(PACKAGED_WORKFLOW.read_text(encoding="utf-8"))["graph"]


def rag_graph(retrieval_mode: str = "dense", **changes: object) -> GraphDescription:
    """Today's graph, its retrieve stage in ``retrieval_mode``, with ``changes``.

    ``changes`` are keywords of ``STAGE_SETTING``. A lexical graph has no embed
    stage.
    """
    graph = packaged_graph()
    stages = graph["stages"]
    stages["retrieve"]["retrieval_mode"] = retrieval_mode
    for keyword, value in changes.items():
        stage, key = STAGE_SETTING[keyword]
        stages[stage][key] = value
    if retrieval_mode == "lexical":  # no vector to compute, so no embed stage
        del stages["embed"]
        graph["edges"] = [
            [n for n in chain if n != "embed"] for chain in graph["edges"]
        ]
    return GraphDescription.model_validate(graph)


def packaged_components() -> dict[str, Any]:
    """The ``components`` block of the workflow file that ships with the code."""
    return yaml.safe_load(PACKAGED_WORKFLOW.read_text(encoding="utf-8"))["components"]


def write_workflow_file(
    path: Path, graph: dict[str, Any], components: dict[str, Any] | None = None
) -> Path:
    """Write a workflow file of ``graph`` and ``components``; return it.

    The components are the packaged file's unless a test gives its own.
    """
    blocks = {"components": components or packaged_components(), "graph": graph}
    path.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    return path
