"""Run labels: the default one a configuration gets, and what a valid one is."""

from __future__ import annotations

import pytest

from rag_load_test.domain.run_label import default_run_label, validate_run_label

# --- default run labels ----------------------------------------------------------

PACKAGED = {
    "embedder": "local",
    "reranker": "local",
    "llm": "api",
    "vector_store": "elasticsearch",
}


def test_the_default_run_label_lists_each_component_with_its_backend() -> None:
    assert default_run_label(PACKAGED) == (
        "embedder=local+reranker=local+llm=api+vector_store=elasticsearch"
    )


def test_the_default_run_label_keeps_the_order_it_is_given() -> None:
    backends = {"vector_store": "fake", "reranker_fr": "served", "llm": "fake"}
    assert default_run_label(backends) == (
        "vector_store=fake+reranker_fr=served+llm=fake"
    )


def test_the_default_run_label_is_a_valid_run_label() -> None:
    label = default_run_label(PACKAGED | {"reranker": "none", "llm": "served"})
    assert validate_run_label(label) == label


@pytest.mark.parametrize(
    "backends",
    [{"re ranker": "local"}, {f"component_{n}": "served" for n in range(10)}],
    ids=["not-header-safe", "too-long"],
)
def test_a_default_run_label_no_header_can_carry_says_to_set_one(
    backends: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="set RAG_RUN_LABEL") as caught:
        default_run_label(backends)

    message = str(caught.value)
    assert f"the default run label {default_label(backends)!r}" in message
    assert "letters, digits and ._+=:- only, 1 to 128 characters" in message


def default_label(backends: dict[str, str]) -> str:
    return "+".join(f"{name}={backend}" for name, backend in backends.items())


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
