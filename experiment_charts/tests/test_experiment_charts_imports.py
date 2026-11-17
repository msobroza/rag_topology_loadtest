"""The charts package takes nothing from the product, read from its files.

It draws the registry of measurements for a reader and is no part of the
product, so it may import neither the toolkit, nor the fake Domino, nor a use
case, and its distribution may depend on none of theirs. The names are those of
the repository's layout (the README's "Where a file goes").
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

from experiment_chart_support import CHARTS_ROOT

CHARTS_SOURCE = CHARTS_ROOT / "src" / "experiment_charts"
# The toolkit, its test double and the RAG use case: the packages of the product.
PRODUCT_PACKAGES = ("query_processing", "fake_domino", "query_processing_rag_usecase")
# Their distributions all start with the toolkit's name.
PRODUCT_DISTRIBUTION_PREFIX = "query-processing"


def absolute_imports(path: Path, package: str) -> list[tuple[int, str]]:
    """``(line, module)`` for every absolute import of ``package`` in ``path``.

    A relative import cannot name another top-level package, so these are all.
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


def test_experiment_charts_imports_nothing_from_the_product_or_the_fake() -> None:
    offending = [
        f"{path.relative_to(CHARTS_SOURCE.parent)}:{line} imports {module}"
        for path in sorted(CHARTS_SOURCE.rglob("*.py"))
        for package in PRODUCT_PACKAGES
        for line, module in absolute_imports(path, package)
    ]
    assert CHARTS_SOURCE.is_dir(), f"{CHARTS_SOURCE} is where the charts package lives"
    assert not offending, f"experiment_charts may import none of them: {offending}"


def test_the_search_finds_an_import_of_each_package_of_the_product(
    tmp_path: Path,
) -> None:
    module = tmp_path / "offender.py"
    module.write_text(
        "import fake_domino.server\n"
        "from query_processing.domain import passages\n"
        "from query_processing_rag_usecase import config\n"
        "from . import chart_table\n"
        "import matplotlib\n",
        encoding="utf-8",
    )
    found = {package: absolute_imports(module, package) for package in PRODUCT_PACKAGES}
    assert found == {
        "query_processing": [(2, "query_processing.domain")],
        "fake_domino": [(1, "fake_domino.server")],
        "query_processing_rag_usecase": [(3, "query_processing_rag_usecase")],
    }


def test_the_distribution_depends_on_nothing_of_the_product() -> None:
    pyproject = (CHARTS_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependencies = tomllib.loads(pyproject)["project"]["dependencies"]
    names = [re.split(r"[^A-Za-z0-9._-]", line, maxsplit=1)[0] for line in dependencies]
    assert "matplotlib" in names
    assert [n for n in names if n.lower().startswith(PRODUCT_DISTRIBUTION_PREFIX)] == []
