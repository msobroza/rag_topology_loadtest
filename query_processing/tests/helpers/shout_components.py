"""What the tests of the ``shouter`` component type share (``shout_extension``).

A workflow file loads the extension by naming it under ``extensions``; these
build what the composition hands an implementation, and a shouter's settings.

Example::

    await open_registered("shouter", shouter_settings(suffix="!"), resources_for("shouter"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from query_processing.config.registered_component_settings import (
    RegisteredComponentSettings,
)
from query_processing.domain.components.component_resources import ComponentResources
from rag_graphs import fixture_components
from shout_extension.shouting import BaseShouter

SHOUT_EXTENSION = "shout_extension"


def shout_graph() -> dict[str, Any]:
    """A graph that searches as the fixture's does, then shouts the question.

    Every stage states its reads and writes. The ``shout`` stage writes the
    answer, so ``/query`` answers with what the shouter said.
    """
    return {
        "stages": {
            "embed_query": {
                "call": "embedder",
                "reads": ["standalone_query"],
                "writes": ["query_embedding"],
            },
            "retrieve": {
                "call": "vector_store",
                "retrieval_mode": "dense",
                "top_k": 4,
                "reads": ["query_embedding", "metadata_filter"],
                "writes": ["candidates"],
            },
            "keep_best": {
                "call": "reranker",
                "top_k": 2,
                "reads": ["standalone_query", "candidates"],
                "writes": ["reranked"],
            },
            "shout": {
                "call": "shouter",
                "reads": ["standalone_query"],
                "writes": ["answer"],
            },
        },
        "edges": [["START", "embed_query", "retrieve", "keep_best", "shout", "END"]],
    }


def write_shout_workflow(
    path: Path,
    extensions: list[str],
    components: dict[str, Any],
    graph: dict[str, Any] | None = None,
) -> Path:
    """Write a workflow file of the fixture's components, ``components`` added; return it."""
    blocks = {
        "extensions": extensions,
        "components": {**fixture_components(), **components},
        "graph": graph or shout_graph(),
    }
    path.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    return path


def resources_for(name: str) -> ComponentResources:
    """The resources of a component that needs nothing from the others."""

    def no_other_component(other: str) -> BaseModel:
        raise ValueError(f"no component named {other!r} in this test")

    return ComponentResources(
        name=name, type_settings=None, component_settings=no_other_component
    )


def shouter_settings(
    backend: str = "local", **local: Any
) -> RegisteredComponentSettings:
    """A ``shouter`` component on ``backend``, its ``local`` block ``local``."""
    entry: dict[str, Any] = {
        "type": BaseShouter.component_type_name,
        "backend": backend,
    }
    if local:
        entry["local"] = local
    return RegisteredComponentSettings.model_validate(entry)
