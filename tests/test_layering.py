"""The layer rule, read from the source with ``ast``: domain imports no other layer;
application and config import only domain (and themselves). And the mock rule:
``fake_domino`` imports nothing from ``rag_load_test``, and only ``deploy/cli.py``
(``--dry-run``) imports ``fake_domino``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "rag_load_test"
FAKE_DOMINO = PACKAGE.parent / "fake_domino"
FAKE_DOMINO_IMPORTERS = {Path("deploy") / "cli.py"}  # relative to PACKAGE
ALLOWED_LAYERS: dict[str, set[str]] = {
    "domain": {"domain"},
    "application": {"application", "domain"},
    "config": {"config", "domain"},
}


def imported_layers(path: Path) -> list[tuple[int, str]]:
    """``(line, sub-package)`` for every ``rag_load_test`` import in ``path``.

    Example::

        imported_layers(PACKAGE / "config" / "settings.py")   # [(21, "domain")]
    """
    layer = path.parent.name
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 1:
            modules = [f"rag_load_test.{layer}"]  # its own sub-package
        elif isinstance(node, ast.ImportFrom) and node.level == 2:
            # ``from ..x import y`` names layer x; ``from .. import x`` names x itself.
            names = [node.module] if node.module else [a.name for a in node.names]
            modules = [f"rag_load_test.{name}" for name in names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        for module in modules:
            parts = module.split(".")
            if parts[0] == "rag_load_test" and len(parts) > 1 and parts[1]:
                found.append((node.lineno, parts[1]))
    return found


@pytest.mark.parametrize("layer", sorted(ALLOWED_LAYERS))
def test_layer_imports_only_the_layers_it_may(layer: str) -> None:
    offending = [
        f"{path.relative_to(PACKAGE)}:{line} imports rag_load_test.{imported}"
        for path in sorted((PACKAGE / layer).glob("*.py"))
        for line, imported in imported_layers(path)
        if imported not in ALLOWED_LAYERS[layer]
    ]
    allowed = ", ".join(sorted(ALLOWED_LAYERS[layer]))
    assert not offending, f"{layer} may import only {allowed}: {offending}"


def absolute_imports(path: Path, package: str) -> list[tuple[int, str]]:
    """``(line, module)`` for every absolute import of ``package`` in ``path``.

    A relative import cannot name another top-level package, so these are all.

    Example::

        absolute_imports(PACKAGE / "deploy" / "cli.py", "fake_domino")
        # [(26, "fake_domino.api_shapes"), ...]
    """
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module or ""]
        else:
            continue
        found += [(node.lineno, m) for m in modules if m.split(".")[0] == package]
    return found


def test_fake_domino_imports_nothing_from_the_product() -> None:
    offending = [
        f"{path.relative_to(FAKE_DOMINO.parent)}:{line} imports {module}"
        for path in sorted(FAKE_DOMINO.rglob("*.py"))
        for line, module in absolute_imports(path, "rag_load_test")
    ]
    assert not offending, f"fake_domino may not import rag_load_test: {offending}"


def test_only_the_deploy_cli_imports_fake_domino() -> None:
    offending = [
        f"{path.relative_to(PACKAGE)}:{line} imports {module}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.relative_to(PACKAGE) not in FAKE_DOMINO_IMPORTERS
        for line, module in absolute_imports(path, "fake_domino")
    ]
    assert not offending, f"only deploy/cli.py may import fake_domino: {offending}"
