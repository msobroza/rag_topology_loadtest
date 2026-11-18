"""The use case's layout rules: its imports, read from its source with ``ast``,
and the size of its directories.

It imports the toolkit through ``query_processing.extension_api`` alone, and
its packages follow the toolkit's layers: a command never imports an
adapter, and the infrastructure imports neither the config nor the adapters.
Its adapters may import its own config, since a class names its settings on
its class line (the plan's departure from the toolkit's layering). No
directory under its ``src/`` and ``tests/`` holds more than 12 modules, with
no exception (rule 9 of the README's "Where a file goes"), measured by the
function the toolkit ships for it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from query_processing.testing import directory_size_findings

USE_CASE_ROOT = Path(__file__).resolve().parents[1]
USE_CASE_SOURCE = USE_CASE_ROOT / "src" / "query_processing_rag_usecase"
MAX_MODULES = 12
PACKAGE = USE_CASE_SOURCE.name
EXTENSION_API = "query_processing.extension_api"
# Package -> the packages of the use case its modules may import.
MAY_IMPORT: dict[str, set[str]] = {
    "config": {"config"},
    "infrastructure": {"infrastructure"},
    "adapters": {"adapters", "infrastructure", "config"},
    "entrypoints": {"entrypoints", "infrastructure", "config"},
}


def imported_modules(path: Path) -> set[str]:
    """Every module the file at ``path`` imports, relative imports resolved."""
    parts = path.relative_to(USE_CASE_SOURCE.parent).with_suffix("").parts
    home = list(parts[:-1])  # the package a relative import starts from
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = home[: len(home) - node.level + 1] if node.level else []
            found.add(".".join([*base, *(node.module or "").split(".")]).strip("."))
    return found


def toolkit_imports_of(source: Path) -> set[str]:
    """The modules of ``query_processing`` the files under ``source`` import."""
    return {
        module
        for path in source.rglob("*.py")
        for module in imported_modules(path)
        if module == "query_processing" or module.startswith("query_processing.")
    }


def test_the_use_case_imports_the_toolkit_only_through_extension_api() -> None:
    assert toolkit_imports_of(USE_CASE_SOURCE) <= {EXTENSION_API}


@pytest.mark.parametrize("package", sorted(MAY_IMPORT))
def test_each_package_imports_only_the_layers_it_may(package: str) -> None:
    for path in (USE_CASE_SOURCE / package).rglob("*.py"):
        imported = {
            module.split(".")[1]
            for module in imported_modules(path)
            if module.startswith(f"{PACKAGE}.")
        }
        assert imported <= MAY_IMPORT[package], f"{path} imports {imported}"


def test_every_package_of_the_use_case_has_its_layer() -> None:
    packages = {
        p.name for p in USE_CASE_SOURCE.iterdir() if (p / "__init__.py").is_file()
    }

    assert packages == set(MAY_IMPORT)


def test_no_directory_of_the_use_case_holds_more_than_twelve_modules() -> None:
    findings = directory_size_findings(USE_CASE_ROOT, limit=MAX_MODULES)
    assert not findings, "\n".join(findings)
