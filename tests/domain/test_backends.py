"""The vector store's Backend names and the defaults beside them."""

from __future__ import annotations

from typing import get_args

from rag_load_test.domain.backends import (
    DEFAULT_GENERATION_BACKEND,
    DEFAULT_VECTOR_STORE_BACKEND,
    VectorStoreBackend,
)


def test_the_default_store_is_one_of_its_backends() -> None:
    assert get_args(VectorStoreBackend) == ("elasticsearch", "chroma", "fake")
    assert DEFAULT_VECTOR_STORE_BACKEND == "elasticsearch"


def test_the_run_label_names_an_llm_off_its_endpoint() -> None:
    assert DEFAULT_GENERATION_BACKEND == "api"
