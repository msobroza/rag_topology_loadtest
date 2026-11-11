"""The topology rule: which backends a topology implies, and run labels."""

from __future__ import annotations

import pytest

from rag_load_test.domain.topology import (
    EMBEDDING_VARIANTS,
    RERANK_VARIANTS,
    TOPOLOGY_BACKENDS,
    TopologyBackends,
    default_run_label,
    resolve_embedding_backend,
    resolve_rerank_backend,
    validate_run_label,
)

TOPOLOGIES = ["monolith", "split-reranker", "split-all"]


# --- the table -------------------------------------------------------------------


def test_the_topology_table() -> None:
    assert {
        "monolith": TopologyBackends(embedding="local", rerank="local"),
        "split-reranker": TopologyBackends(embedding="local", rerank="api"),
        "split-all": TopologyBackends(embedding="api", rerank="api"),
    } == TOPOLOGY_BACKENDS


def test_the_variants_load_no_model_of_ours() -> None:
    assert EMBEDDING_VARIANTS == ("api", "fake")
    assert RERANK_VARIANTS == ("api", "fake", "none")


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


# --- default run labels ----------------------------------------------------------


def label_of(topology: str = "monolith", **choices: str) -> str:
    defaults = {
        "embedding": TOPOLOGY_BACKENDS[topology].embedding,
        "rerank": TOPOLOGY_BACKENDS[topology].rerank,
        "store": "elasticsearch",
        "generation": "api",
    }
    return default_run_label(topology, **(defaults | choices))


def test_default_run_label_is_the_topology_when_nothing_differs() -> None:
    assert label_of("monolith") == "monolith"
    assert label_of("split-all") == "split-all"


def test_default_run_label_appends_a_skipped_rerank() -> None:
    assert label_of("monolith", rerank="none") == "monolith+rerank=none"


def test_default_run_label_appends_a_fake_generation() -> None:
    assert label_of("split-all", generation="fake") == "split-all+generation=fake"


def test_default_run_label_names_an_api_where_the_topology_runs_local() -> None:
    assert label_of("monolith", embedding="api") == "monolith+embedding=api"


def test_default_run_label_lists_every_choice_in_a_fixed_order() -> None:
    label = label_of(
        "monolith",
        embedding="fake",
        rerank="fake",
        store="fake",
        generation="fake",
        retrieval="hybrid",
    )

    assert label == (
        "monolith+embedding=fake+rerank=fake+store=fake+generation=fake"
        "+retrieval=hybrid"
    )


def test_default_run_label_appends_a_retrieval_mode_that_is_not_dense() -> None:
    assert label_of(store="chroma", retrieval="dense") == "monolith+store=chroma"
    assert label_of(store="chroma", retrieval="hybrid") == (
        "monolith+store=chroma+retrieval=hybrid"
    )


def test_the_longest_default_run_label_is_a_valid_run_label() -> None:
    label = label_of(
        "split-reranker",
        embedding="fake",
        rerank="none",
        store="chroma",
        generation="fake",
        retrieval="lexical",
    )

    assert validate_run_label(label) == label


# --- run label validation --------------------------------------------------------


@pytest.mark.parametrize(
    "label", ["split-reranker-fastapi", "a", "gpu.t4+rerank=none:v2_1", "x" * 128]
)
def test_valid_run_labels_pass_unchanged(label: str) -> None:
    assert validate_run_label(label) == label


@pytest.mark.parametrize("label", ["", "x" * 129, "has space", "new\nline", "é"])
def test_invalid_run_labels_name_the_value_and_the_shape(label: str) -> None:
    with pytest.raises(ValueError, match="RAG_RUN_LABEL=") as caught:
        validate_run_label(label)

    message = str(caught.value)
    assert f"RAG_RUN_LABEL={label!r}" in message
    assert "letters, digits and ._+=:-" in message
    assert "1 to 128 characters" in message
