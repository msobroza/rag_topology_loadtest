"""The core imports only the standard library, its own package and the few
third-party packages it is allowed, read from its files with ``ast``.

The core is data that every toolkit installs (ADR 0014), so it may import no
toolkit and no use case: any package of the repository but its own is refused,
as any third-party package outside ``ALLOWED_THIRD_PARTY`` is. The test reads
the files and imports none of them, so it starts nothing of the toolkit.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE = CORE_ROOT / "src" / "query_processing_core"
# The third-party packages the core may import, by top-level module name:
# Pydantic, and doclang's core (doclang-domain-core, whose namespace is
# doclang_domain) for the vector-store record. One module of the core names
# doclang (the toolkit's tests/structure/test_core_rules.py).
ALLOWED_THIRD_PARTY = frozenset({"pydantic", "doclang_domain"})


def imported_top_levels(source: str) -> list[tuple[int, str]]:
    """``(line, top-level package)`` for every absolute import in ``source``.

    A relative import stays inside the core's own package, so it is not listed.

    Example:
        >>> imported_top_levels("import json\\nfrom pydantic import BaseModel")
        [(1, 'json'), (2, 'pydantic')]
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name.split(".")[0]) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.append((node.lineno, (node.module or "").split(".")[0]))
    return sorted(found)


def refused_imports(where: str, source: str) -> list[str]:
    """What ``source``, the file at ``where``, imports that the core may not.

    Example:
        >>> refused_imports("a.py", "from query_processing.domain import errors")
        ['a.py:1 imports query_processing']
    """
    allowed = sys.stdlib_module_names | ALLOWED_THIRD_PARTY | {CORE_SOURCE.name}
    return [
        f"{where}:{line} imports {package}"
        for line, package in imported_top_levels(source)
        if package not in allowed
    ]


def test_the_import_rule_on_a_literal_source() -> None:
    source = (
        "from __future__ import annotations\n"
        "import json\n"
        "from pydantic import BaseModel\n"
        "from query_processing_core.retrieval import metadata_filter\n"
        "from . import retrieval\n"
        "from query_processing.domain.errors import RagDependencyError\n"
        "import query_processing_rag_usecase.config\n"
        "import fastapi\n"
    )
    assert refused_imports("m.py", source) == [
        "m.py:6 imports query_processing",
        "m.py:7 imports query_processing_rag_usecase",
        "m.py:8 imports fastapi",
    ]


def test_the_core_imports_only_the_standard_library_and_what_it_is_allowed() -> None:
    files = sorted(CORE_SOURCE.rglob("*.py"))
    refused = [
        refusal
        for path in files
        for refusal in refused_imports(
            str(path.relative_to(CORE_ROOT)), path.read_text(encoding="utf-8")
        )
    ]
    assert files, f"no module was read under {CORE_SOURCE}: the walk is wrong"
    allowed = sorted(ALLOWED_THIRD_PARTY)
    assert not refused, f"the core may import the stdlib and {allowed}: {refused}"
