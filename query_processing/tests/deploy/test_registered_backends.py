"""The backends of each component type, as qp-deploy reads them without importing.

qp-deploy may not import the composition root, so it takes a type's backends
from the names of its opener modules. These tests hold that view to what the
openers register: a backend is what is registered, not a list written for the
deploy tool.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

from query_processing.composition.openers.component_openers import OPENERS_BY_TYPE
from query_processing.deploy.registered_backends import (
    is_registered_type,
    registered_backends,
)


@pytest.mark.parametrize("component_type", sorted(OPENERS_BY_TYPE))
def test_the_backends_read_are_the_backends_registered(component_type: str) -> None:
    assert registered_backends(component_type) == (
        OPENERS_BY_TYPE[component_type].names()
    )


def test_neither_the_vector_store_nor_the_llm_runs_locally() -> None:
    assert "local" not in registered_backends("vector_store")
    assert "local" not in registered_backends("llm")
    assert "local" in registered_backends("reranker")


@pytest.mark.parametrize(
    ("component_type", "backends"),
    [
        # served is built by the toolkit for the shouter and the murmurer, and
        # the whisperer names a served class of its own: once either way.
        ("shouter", ("fake", "local", "served")),
        ("whisperer", ("local", "served")),
        ("murmurer", ("fake", "served")),
    ],
)
def test_a_registered_type_has_its_class_line_backends_and_served(
    component_type: str, backends: tuple[str, ...]
) -> None:
    import_module("shout_extension.whispering")  # as a workflow file loads it
    assert is_registered_type(component_type)
    assert registered_backends(component_type) == backends


def test_a_type_with_openers_or_unregistered_is_no_registered_type() -> None:
    import_module("query_processing.adapters.llm.base_llm")  # registers llm
    assert not is_registered_type("llm")
    assert not is_registered_type("no_such_type")
    assert registered_backends("no_such_type") == ()


def test_a_new_opener_module_is_a_new_backend_of_its_type(tmp_path: Path) -> None:
    for name in ("llm_on_api.py", "llm_on_echo.py", "reranker_on_local.py"):
        (tmp_path / name).write_text("")
    assert registered_backends("llm", tmp_path) == ("api", "echo")
    assert registered_backends("embedder", tmp_path) == ()
