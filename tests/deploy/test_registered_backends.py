"""The backends of each component type, as rag-deploy reads them without importing.

rag-deploy may not import the composition root, so it takes a type's backends
from the names of its opener modules. These tests hold that view to what the
openers register: a backend is what is registered, not a list written for the
deploy tool.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_load_test.composition.openers.component_openers import OPENERS_BY_TYPE
from rag_load_test.deploy.registered_backends import registered_backends


@pytest.mark.parametrize("component_type", sorted(OPENERS_BY_TYPE))
def test_the_backends_read_are_the_backends_registered(component_type: str) -> None:
    assert registered_backends(component_type) == (
        OPENERS_BY_TYPE[component_type].names()
    )


def test_neither_the_vector_store_nor_the_llm_runs_locally() -> None:
    assert "local" not in registered_backends("vector_store")
    assert "local" not in registered_backends("llm")
    assert "local" in registered_backends("reranker")


def test_a_new_opener_module_is_a_new_backend_of_its_type(tmp_path: Path) -> None:
    for name in ("llm_on_api.py", "llm_on_echo.py", "reranker_on_local.py"):
        (tmp_path / name).write_text("")
    assert registered_backends("llm", tmp_path) == ("api", "echo")
    assert registered_backends("embedder", tmp_path) == ()
