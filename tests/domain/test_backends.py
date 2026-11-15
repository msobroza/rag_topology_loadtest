"""The default Backends the run label is measured against."""

from __future__ import annotations

from rag_load_test.composition.openers.component_openers import VECTOR_STORE_OPENERS
from rag_load_test.domain.backends import (
    DEFAULT_GENERATION_BACKEND,
    DEFAULT_VECTOR_STORE_BACKEND,
)


def test_the_default_store_is_one_of_its_backends() -> None:
    assert DEFAULT_VECTOR_STORE_BACKEND == "elasticsearch"
    assert DEFAULT_VECTOR_STORE_BACKEND in VECTOR_STORE_OPENERS.names()


def test_the_run_label_names_an_llm_off_its_endpoint() -> None:
    assert DEFAULT_GENERATION_BACKEND == "api"
