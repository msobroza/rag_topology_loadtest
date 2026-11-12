"""The topology rule: which backends a topology implies, and what contradicts it."""

from __future__ import annotations

import pytest

from rag_load_test.domain.topology import (
    TOPOLOGY_BACKENDS,
    TopologyBackends,
    resolve_embedding_backend,
    resolve_rerank_backend,
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
def test_unset_backends_take_the_topology_value(
    topology: str, embedding: str, rerank: str
) -> None:
    assert resolve_embedding_backend(topology, None) == embedding
    assert resolve_rerank_backend(topology, None) == rerank


def test_an_explicit_backend_may_repeat_the_topology_value() -> None:
    assert resolve_embedding_backend("split-all", "api") == "api"
    assert resolve_rerank_backend("monolith", "local") == "local"


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("backend", ["api", "fake"])
def test_embedding_variants_are_allowed_in_every_topology(
    topology: str, backend: str
) -> None:
    # An endpoint or a fake loads no model of ours, so where ours run does not
    # constrain it.
    assert resolve_embedding_backend(topology, backend) == backend


@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("backend", ["api", "fake", "none"])
def test_rerank_variants_are_allowed_in_every_topology(
    topology: str, backend: str
) -> None:
    assert resolve_rerank_backend(topology, backend) == backend


# --- contradictions --------------------------------------------------------------


def test_a_local_embedding_contradicts_split_all() -> None:
    with pytest.raises(ValueError, match="RAG_EMBEDDING_BACKEND='local'") as caught:
        resolve_embedding_backend("split-all", "local")

    message = str(caught.value)
    assert "RAG_TOPOLOGY='split-all'" in message
    assert "only 'api' or 'fake' for embedding" in message
    assert (
        "set RAG_TOPOLOGY=monolith or RAG_TOPOLOGY=split-reranker, "
        "or drop RAG_EMBEDDING_BACKEND"
    ) in message


@pytest.mark.parametrize("topology", ["split-reranker", "split-all"])
def test_a_local_rerank_contradicts_a_split_topology(topology: str) -> None:
    with pytest.raises(ValueError, match="RAG_RERANK_BACKEND='local'") as caught:
        resolve_rerank_backend(topology, "local")

    message = str(caught.value)
    assert f"RAG_TOPOLOGY={topology!r}" in message
    assert "only 'api', 'fake' or 'none' for rerank" in message
    assert "set RAG_TOPOLOGY=monolith, or drop RAG_RERANK_BACKEND" in message


def test_none_is_not_an_embedding_backend() -> None:
    with pytest.raises(ValueError, match="RAG_EMBEDDING_BACKEND='none'") as caught:
        resolve_embedding_backend("monolith", "none")

    message = str(caught.value)
    assert "'local', 'api' or 'fake'" in message
    assert message.endswith("drop RAG_EMBEDDING_BACKEND")
