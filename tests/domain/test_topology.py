"""The topology rule: which backends a topology implies, and what it moves."""

from __future__ import annotations

import pytest

from rag_load_test.domain.topology import (
    TOPOLOGY_BACKENDS,
    TopologyBackends,
    placed_backend,
)

TOPOLOGIES = ["monolith", "split-reranker", "split-all"]


# --- the table -------------------------------------------------------------------


def test_the_topology_table() -> None:
    assert {
        "monolith": TopologyBackends(embedding="local", rerank="local"),
        "split-reranker": TopologyBackends(embedding="local", rerank="api"),
        "split-all": TopologyBackends(embedding="api", rerank="api"),
    } == TOPOLOGY_BACKENDS


# --- backends per topology -------------------------------------------------------


@pytest.mark.parametrize(
    ("topology", "embedding", "rerank"),
    [
        ("monolith", "local", "local"),
        ("split-reranker", "local", "api"),
        ("split-all", "api", "api"),
    ],
)
def test_a_local_component_takes_the_topology_value(
    topology: str, embedding: str, rerank: str
) -> None:
    assert placed_backend(topology, "embedding", "local") == embedding
    assert placed_backend(topology, "rerank", "local") == rerank


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("backend", ["api", "fake", "none", "gpu"])
def test_any_other_backend_is_kept_in_every_topology(
    topology: str, backend: str
) -> None:
    # An endpoint, a fake or no rerank loads no model of ours, so where ours run
    # does not constrain it; an unknown name is refused by the openers.
    assert placed_backend(topology, "embedding", backend) == backend
    assert placed_backend(topology, "rerank", backend) == backend
