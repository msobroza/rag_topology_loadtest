"""The topology rule: which backends a topology implies, and run labels."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_load_test.config.settings import RagSettings
from rag_load_test.domain.topology import (
    default_run_label,
    resolve_embedder_backend,
    resolve_reranker_backend,
    validate_run_label,
)

SettingsFactory = Callable[..., RagSettings]


# --- backends per topology -----------------------------------------------------


@pytest.mark.parametrize(
    ("topology", "embedder", "reranker"),
    [
        ("monolith", "local", "local"),
        ("split-reranker", "local", "ovms"),
        ("split-all", "ovms", "ovms"),
    ],
)
def test_unset_backends_take_the_topology_value(
    topology: str, embedder: str, reranker: str
) -> None:
    assert resolve_embedder_backend(topology, None) == embedder  # type: ignore[arg-type]
    assert resolve_reranker_backend(topology, None) == reranker  # type: ignore[arg-type]


def test_an_explicit_backend_may_repeat_the_topology_value() -> None:
    assert resolve_embedder_backend("split-all", "ovms") == "ovms"
    assert resolve_reranker_backend("monolith", "local") == "local"


@pytest.mark.parametrize("topology", ["monolith", "split-reranker", "split-all"])
def test_fake_and_none_are_allowed_in_every_topology(topology: str) -> None:
    assert resolve_embedder_backend(topology, "fake") == "fake"  # type: ignore[arg-type]
    assert resolve_reranker_backend(topology, "fake") == "fake"  # type: ignore[arg-type]
    assert resolve_reranker_backend(topology, "none") == "none"  # type: ignore[arg-type]


@pytest.mark.parametrize("topology", ["monolith", "split-reranker", "split-all"])
def test_hosted_apis_are_allowed_in_every_topology(topology: str) -> None:
    # A hosted API runs no model of ours, so where ours run does not constrain it.
    assert resolve_embedder_backend(topology, "openai") == "openai"  # type: ignore[arg-type]
    assert resolve_reranker_backend(topology, "cohere") == "cohere"  # type: ignore[arg-type]
    assert resolve_reranker_backend(topology, "openai") == "openai"  # type: ignore[arg-type]


def test_cohere_is_not_an_embedder_backend() -> None:
    with pytest.raises(ValueError, match="RAG_EMBEDDER_BACKEND='cohere'"):
        resolve_embedder_backend("monolith", "cohere")  # type: ignore[arg-type]


# --- contradictions --------------------------------------------------------------


def test_an_ovms_embedder_contradicts_the_monolith() -> None:
    with pytest.raises(ValueError, match="RAG_EMBEDDER_BACKEND='ovms'") as caught:
        resolve_embedder_backend("monolith", "ovms")

    message = str(caught.value)
    assert "RAG_EMBEDDER_BACKEND='ovms'" in message
    assert "RAG_TOPOLOGY='monolith'" in message
    assert "'local', 'openai' or 'fake'" in message
    assert "set RAG_TOPOLOGY=split-all, or drop RAG_EMBEDDER_BACKEND" in message


def test_an_ovms_reranker_contradicts_the_monolith_and_names_both_fixes() -> None:
    with pytest.raises(ValueError, match="RAG_RERANKER_BACKEND='ovms'") as caught:
        resolve_reranker_backend("monolith", "ovms")

    message = str(caught.value)
    assert "RAG_RERANKER_BACKEND='ovms'" in message
    assert "'local', 'cohere', 'openai', 'fake' or 'none'" in message
    assert (
        "set RAG_TOPOLOGY=split-reranker or RAG_TOPOLOGY=split-all, "
        "or drop RAG_RERANKER_BACKEND"
    ) in message


def test_a_local_reranker_contradicts_a_split_topology() -> None:
    with pytest.raises(ValueError, match="RAG_RERANKER_BACKEND='local'") as caught:
        resolve_reranker_backend("split-all", "local")

    assert "RAG_TOPOLOGY='split-all'" in str(caught.value)
    assert "set RAG_TOPOLOGY=monolith" in str(caught.value)


def test_a_local_embedder_contradicts_split_all() -> None:
    with pytest.raises(ValueError, match="RAG_TOPOLOGY='split-all'") as caught:
        resolve_embedder_backend("split-all", "local")

    assert "RAG_TOPOLOGY=monolith or RAG_TOPOLOGY=split-reranker" in str(caught.value)


def test_none_is_not_an_embedder_backend() -> None:
    with pytest.raises(ValueError, match="RAG_EMBEDDER_BACKEND='none'") as caught:
        resolve_embedder_backend("monolith", "none")  # type: ignore[arg-type]

    assert "drop RAG_EMBEDDER_BACKEND" in str(caught.value)


# --- default run labels ----------------------------------------------------------


def test_default_run_label_is_the_topology_when_nothing_differs() -> None:
    label = default_run_label(
        "monolith",
        embedder="local",
        reranker="local",
        store="elasticsearch",
        llm="openai",
    )

    assert label == "monolith"


def test_default_run_label_appends_a_skipped_reranker() -> None:
    label = default_run_label(
        "monolith",
        embedder="local",
        reranker="none",
        store="elasticsearch",
        llm="openai",
    )

    assert label == "monolith+reranker=none"


def test_default_run_label_appends_a_fake_llm() -> None:
    label = default_run_label(
        "split-all", embedder="ovms", reranker="ovms", store="elasticsearch", llm="fake"
    )

    assert label == "split-all+llm=fake"


def test_default_run_label_lists_every_port_in_a_fixed_order() -> None:
    label = default_run_label(
        "monolith", embedder="fake", reranker="fake", store="fake", llm="fake"
    )

    assert label == "monolith+embedder=fake+reranker=fake+store=fake+llm=fake"


def test_the_longest_default_run_label_is_a_valid_run_label() -> None:
    label = default_run_label(
        "split-reranker", embedder="fake", reranker="none", store="fake", llm="fake"
    )

    assert validate_run_label(label) == label


def test_default_run_label_appends_a_retrieval_mode_that_is_not_dense() -> None:
    def label(retrieval: str) -> str:
        return default_run_label(
            "monolith",
            embedder="local",
            reranker="local",
            store="chroma",
            llm="openai",
            retrieval=retrieval,
        )

    assert label("dense") == "monolith+store=chroma"
    assert label("hybrid") == "monolith+store=chroma+retrieval=hybrid"


def test_the_longest_default_run_label_with_hosted_backends_is_valid() -> None:
    label = default_run_label(
        "split-reranker",
        embedder="openai",
        reranker="cohere",
        store="elasticsearch",
        llm="fake",
        retrieval="lexical",
    )

    assert validate_run_label(label) == label


# --- run label validation --------------------------------------------------------


@pytest.mark.parametrize(
    "label", ["split-reranker-fastapi", "a", "gpu.t4+reranker=none:v2_1", "x" * 128]
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
