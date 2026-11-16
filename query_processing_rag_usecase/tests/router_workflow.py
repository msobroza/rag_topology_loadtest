"""A workflow file of the router alone, every component on its fake backend.

The router's tests train pipelines, and the use case's ``workflow.yaml`` names
its pipeline file in this directory, so they write their own file in
``tmp_path``: the router's embedder (Solon's alias and
query prefix, on the fake backend), the router listing the two routes of this
directory's ``routes.yaml``, and a two-stage graph that embeds then routes.
A pipeline the tests train is written beside it, never in this directory.

Example::

    settings = settings_on_fakes(tmp_path, monkeypatch)
    manifest = await train_router(settings, "router", allow_fake_embedder=True)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from query_processing.extension_api import WorkflowSettings

USE_CASE_DIR = Path(__file__).resolve().parents[1]
ROUTES = ["self_contained", "leans_on_history"]
PIPELINE_FILE = "router_pipeline.joblib"


def router_components(tmp_path: Path, **local: Any) -> dict[str, Any]:
    """The router and its embedder, on fake backends, the ``local`` block given."""
    block = {
        "pipeline_file": str(tmp_path / PIPELINE_FILE),
        "n_neighbors": 3,
        "metric": "cosine",
        **local,
    }
    return {
        "router_embedder": {
            "type": "embedder",
            "backend": "fake",
            "source_id": "OrdalieTech/Solon-embeddings-large-0.1",
            "query_prefix": "query : ",
        },
        "router": {
            "backend": "fake",
            "embedder": "router_embedder",
            "routes": ROUTES,
            "local": block,
        },
    }


def router_graph() -> dict[str, Any]:
    """Embed the message with the router's embedder, then route it."""
    return {
        "stages": {
            "embed_for_route": {
                "call": "router_embedder",
                "reads": ["conversational_query"],
                "writes": ["routing_embedding"],
            },
            "route": {
                "call": "router",
                "reads": ["routing_embedding"],
                "writes": ["route"],
            },
        },
        "edges": [["START", "embed_for_route", "route", "END"]],
    }


def write_router_workflow(tmp_path: Path, **local: Any) -> Path:
    """Write the router's workflow file in ``tmp_path``; return it."""
    path = tmp_path / "workflow.yaml"
    blocks = {
        "extensions": ["query_processing_rag_usecase"],
        "components": router_components(tmp_path, **local),
        "graph": router_graph(),
    }
    path.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    return path


def settings_on_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **local: Any
) -> WorkflowSettings:
    """The settings of the router's workflow file, read with this directory's routes."""
    for name in [n for n in os.environ if n.startswith(("QP_", "DOMINO_"))]:
        monkeypatch.delenv(name)
    monkeypatch.setenv(
        "QP_WORKFLOW_FILE", str(write_router_workflow(tmp_path, **local))
    )
    monkeypatch.setenv("QP_ROUTES_FILE", str(USE_CASE_DIR / "routes.yaml"))
    monkeypatch.chdir(tmp_path)
    return WorkflowSettings(_env_file=None)  # type: ignore[call-arg]
