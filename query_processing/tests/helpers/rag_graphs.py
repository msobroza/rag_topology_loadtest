"""The tests' workflow file, and its graph with some stages' settings changed.

The toolkit ships no workflow file and its tests read no use case's: they run
on ``fixture_workflow.yaml``, beside this module, a four-stage retrieval graph
over made-up models, so a test names only what it changes. A lexical graph has
no embed stage, as a lexical workflow file has none: its first chain starts at
retrieve. The retrieve stage reads what its retrieval mode needs. A test hands the graph to the settings as ``graph=``, which replaces
the file's whole graph.

Example::

    settings_factory(graph=rag_graph("hybrid", retrieve_top_k=3, rerank_top_k=3))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from query_processing.domain.graph_description import GraphDescription

# The workflow file every test reads unless it names another (conftest.py).
FIXTURE_WORKFLOW = Path(__file__).with_name("fixture_workflow.yaml")
# Its prompts, which every test reads unless it names another file (conftest.py).
FIXTURE_PROMPTS = Path(__file__).with_name("fixture_prompts.yaml")
# Each keyword of rag_graph: the stage it sets and the key of that stage.
STAGE_SETTING = {
    "retrieve_top_k": ("retrieve", "top_k"),
    "rrf_k": ("retrieve", "rrf_k"),
    "rerank_top_k": ("rerank", "top_k"),
    "prompt": ("generate", "prompt"),
}
# What the retrieve stage reads in each retrieval mode: a vector, words, or both,
# then the metadata filter.
RETRIEVE_READS = {
    "dense": ["query_embedding", "metadata_filter"],
    "lexical": ["standalone_query", "metadata_filter"],
    "hybrid": ["query_embedding", "standalone_query", "metadata_filter"],
}


def fixture_graph() -> dict[str, Any]:
    """The ``graph`` block of the tests' workflow file, as read."""
    return yaml.safe_load(FIXTURE_WORKFLOW.read_text(encoding="utf-8"))["graph"]


def rag_graph(retrieval_mode: str = "dense", **changes: object) -> GraphDescription:
    """The fixture's graph, its retrieve stage in ``retrieval_mode``, with ``changes``.

    ``changes`` are keywords of ``STAGE_SETTING``. The retrieve stage reads
    what ``retrieval_mode`` needs, and a lexical graph has no embed stage.
    """
    graph = fixture_graph()
    stages = graph["stages"]
    stages["retrieve"]["retrieval_mode"] = retrieval_mode
    stages["retrieve"]["reads"] = RETRIEVE_READS[retrieval_mode]
    for keyword, value in changes.items():
        stage, key = STAGE_SETTING[keyword]
        stages[stage][key] = value
    if retrieval_mode == "lexical":  # no vector to compute, so no embed stage
        del stages["embed"]
        graph["edges"] = [
            [n for n in chain if n != "embed"] for chain in graph["edges"]
        ]
    return GraphDescription.model_validate(graph)


def fixture_components() -> dict[str, Any]:
    """The ``components`` block of the tests' workflow file."""
    return yaml.safe_load(FIXTURE_WORKFLOW.read_text(encoding="utf-8"))["components"]


def write_workflow_file(
    path: Path, graph: dict[str, Any], components: dict[str, Any] | None = None
) -> Path:
    """Write a workflow file of ``graph`` and ``components``; return it.

    The components are the fixture's unless a test gives its own.
    """
    blocks = {"components": components or fixture_components(), "graph": graph}
    path.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    return path
