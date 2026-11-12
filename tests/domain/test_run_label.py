"""Run labels: the default one a configuration gets, and what a valid one is."""

from __future__ import annotations

import pytest

from rag_load_test.domain.run_label import default_run_label, validate_run_label
from rag_load_test.domain.topology import TOPOLOGY_BACKENDS

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
