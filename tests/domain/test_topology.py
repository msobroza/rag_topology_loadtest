"""The topology rule: which backends a topology implies, and what it moves."""

from __future__ import annotations

import pytest

from rag_load_test.domain.topology import (
    TOPOLOGY_BACKENDS,
    TopologyBackends,
    placed_backend,
    topology_that_moves_nothing,
)

TOPOLOGIES = ["monolith", "split-reranker", "split-all"]


# --- the table -------------------------------------------------------------------


def test_the_topology_table() -> None:
    assert {
        "monolith": TopologyBackends(embedding="local", rerank="local"),
        "split-reranker": TopologyBackends(embedding="local", rerank="served"),
        "split-all": TopologyBackends(embedding="served", rerank="served"),
    } == TOPOLOGY_BACKENDS


# --- backends per topology -------------------------------------------------------


@pytest.mark.parametrize(
    ("topology", "embedding", "rerank"),
    [
        ("monolith", "local", "local"),
        ("split-reranker", "local", "served"),
        ("split-all", "served", "served"),
    ],
)
def test_a_local_component_takes_the_topology_value(
    topology: str, embedding: str, rerank: str
) -> None:
    assert placed_backend(topology, "embedding", "local") == embedding
    assert placed_backend(topology, "rerank", "local") == rerank


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("backend", ["served", "api", "fake", "none", "gpu"])
def test_any_other_backend_is_kept_in_every_topology(
    topology: str, backend: str
) -> None:
    # A model server, an endpoint, a fake or no rerank loads no model of ours
    # here, so where ours run does not constrain it; an unknown name is refused
    # by the openers.
    assert placed_backend(topology, "embedding", backend) == backend
    assert placed_backend(topology, "rerank", backend) == backend


# --- the topology the workflow app is told -----------------------------------------


@pytest.mark.parametrize(
    ("alias", "backends", "told"),
    [
        # The three aliases of the deploy settings example.
        ("monolith", {"embedder": "local", "reranker": "local"}, "monolith"),
        (
            "split-reranker",
            {"embedder": "local", "reranker": "served"},
            "split-reranker",
        ),
        ("split-all", {"embedder": "served", "reranker": "served"}, "split-all"),
        # The rule keeps every backend but local; a component it does not place passes.
        (
            "split-all",
            {"embedder": "api", "reranker": "none", "llm": "api"},
            "split-all",
        ),
        # A name the workflow app does not know.
        ("split-embedder", {"embedder": "served", "reranker": "local"}, "monolith"),
        # A known name whose rule would move a local component the alias keeps.
        ("split-all", {"embedder": "served", "reranker": "local"}, "monolith"),
        ("split-reranker", {"embedder": "local", "reranker": "local"}, "monolith"),
    ],
)
def test_the_workflow_app_is_told_a_topology_whose_rule_moves_nothing(
    alias: str, backends: dict[str, str], told: str
) -> None:
    assert topology_that_moves_nothing(alias, backends) == told
