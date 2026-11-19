"""An orchestrator's library is imported by its adapter alone (ADR 0010).

Read from the source with ``ast``: no module outside the adapter's directory
imports the library, so the application and the domain import none.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from repo_paths import TOOLKIT_ROOT

PACKAGE = TOOLKIT_ROOT / "src" / "query_processing"
# An orchestrator's library -> the one directory that may import it, its adapter.
ORCHESTRATOR_LIBRARY_HOMES = {
    "langgraph": PACKAGE / "adapters" / "orchestrator" / "langgraph",
}


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


@pytest.mark.parametrize("library", sorted(ORCHESTRATOR_LIBRARY_HOMES))
def test_an_orchestrators_library_is_imported_by_its_adapter_alone(
    library: str,
) -> None:
    home = ORCHESTRATOR_LIBRARY_HOMES[library]
    importers = [
        path
        for path in sorted(PACKAGE.rglob("*.py"))
        if imports_library(path.read_text(encoding="utf-8"), library)
    ]
    offending = [_relative(path) for path in importers if home not in path.parents]
    assert importers, f"nothing imports {library}: is {_relative(home)} its adapter?"
    assert not offending, f"only {_relative(home)} may import {library}: {offending}"


def _relative(path: Path) -> str:
    return path.relative_to(PACKAGE).as_posix()
