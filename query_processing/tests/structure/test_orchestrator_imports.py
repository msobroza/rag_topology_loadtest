"""An orchestrator's library is imported by its adapter alone (ADR 0010).

Read from the source with ``ast``: no module outside the adapter's directory
imports the library, so the application and the domain import none. The
libraries are the registered orchestrators' names, read when the tests are
collected: an orchestrator's directory of ``adapters/orchestrator/`` is named
after the library it imports, so a third one is checked with no edit here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from query_processing.testing import registered_orchestrators
from repo_paths import TOOLKIT_ROOT

PACKAGE = TOOLKIT_ROOT / "src" / "query_processing"
# Each orchestrator's directory, named after its library: the one that may import it.
ORCHESTRATORS_HOME = PACKAGE / "adapters" / "orchestrator"


def imports_library(source: str, library: str) -> bool:
    """Whether ``source`` imports ``library`` or a module of it.

    Example::

        imports_library("from langgraph.graph import StateGraph", "langgraph")  # True
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module or ""]
        else:
            continue
        if any(module.split(".")[0] == library for module in modules):
            return True
    return False


def test_the_library_reader_on_literal_sources() -> None:
    assert imports_library("from langgraph.graph import StateGraph", "langgraph")
    assert imports_library("import langgraph", "langgraph")
    assert not imports_library("from .langgraph import x", "langgraph")
    assert not imports_library("import langgraph_sdk", "langgraph")
    assert imports_library("from pydantic_graph import Graph", "pydantic_graph")
    assert not imports_library("from .pydantic_graph import x", "pydantic_graph")


@pytest.mark.parametrize("library", registered_orchestrators())
def test_an_orchestrators_library_is_imported_by_its_adapter_alone(
    library: str,
) -> None:
    home = ORCHESTRATORS_HOME / library
    importers = [
        path
        for path in sorted(PACKAGE.rglob("*.py"))
        if imports_library(path.read_text(encoding="utf-8"), library)
    ]
    offending = [_relative(path) for path in importers if home not in path.parents]
    assert importers, (
        f"nothing under {_relative(home)} imports a library named {library!r}: "
        "an orchestrator's directory is named after the library it imports"
    )
    assert not offending, f"only {_relative(home)} may import {library}: {offending}"


def _relative(path: Path) -> str:
    return path.relative_to(PACKAGE).as_posix()
