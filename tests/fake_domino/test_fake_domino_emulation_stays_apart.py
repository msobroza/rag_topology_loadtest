"""The emulation stays a removable add-on of the core fake Domino.

Rule 1: no core module (``src/fake_domino/*.py``) imports ``fake_domino.emulation``,
not even inside a function. Rule 2: ``jwt`` and ``cryptography`` are imported only
under ``src/fake_domino/emulation/``. Also: the core's tests import nothing of it.
"""

from __future__ import annotations

import ast
from pathlib import Path

from repo_paths import REPO_ROOT

SRC = REPO_ROOT / "src"
CORE = SRC / "fake_domino"
EMULATION = CORE / "emulation"
EMULATION_PACKAGE = "fake_domino.emulation"
EMULATION_ONLY_LIBRARIES = ("jwt", "cryptography")


def imported_modules(path: Path, package: str) -> set[str]:
    """Every module ``path`` imports, anywhere in it, relative imports resolved."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            parts = package.split(".")
            base = parts[: len(parts) - node.level + 1] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def is_under(module: str, root: str) -> bool:
    return module == root or module.startswith(f"{root}.")


def test_the_measure_resolves_relative_imports(tmp_path: Path) -> None:
    source = tmp_path / "m.py"
    source.write_text("def f():\n    from .emulation import server\n")
    assert "fake_domino.emulation.server" in imported_modules(source, "fake_domino")


def test_no_core_module_imports_the_emulation() -> None:
    offending = [
        f"{path.name}: {module}"
        for path in sorted(CORE.glob("*.py"))
        for module in imported_modules(path, "fake_domino")
        if is_under(module, EMULATION_PACKAGE)
    ]
    assert not offending, f"core modules import the emulation: {offending}"


def test_jwt_and_cryptography_are_imported_only_by_the_emulation() -> None:
    offending = []
    for path in sorted(SRC.rglob("*.py")):
        if path.is_relative_to(EMULATION):
            continue
        package = ".".join(path.relative_to(SRC).parts[:-1])
        offending += [
            f"{path.relative_to(SRC)}: {module}"
            for module in imported_modules(path, package)
            if any(is_under(module, lib) for lib in EMULATION_ONLY_LIBRARIES)
        ]
    assert not offending, f"emulation-only libraries imported outside it: {offending}"


def test_the_core_tests_import_nothing_of_the_emulation() -> None:
    tests = REPO_ROOT / "tests" / "fake_domino"
    core_tests = [
        path
        for path in sorted(tests.glob("*.py"))
        if not path.name.startswith("test_fake_domino_emulation_")
    ]
    offending = [
        f"{path.name}: {module}"
        for path in core_tests
        for module in imported_modules(path, "")
        if is_under(module, EMULATION_PACKAGE)
        or module.startswith("test_fake_domino_emulation_")
    ]
    assert core_tests
    assert not offending, f"core tests import the emulation: {offending}"
