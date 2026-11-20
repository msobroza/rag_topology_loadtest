"""One module of the repository's distributions imports doclang (ADR 0014).

doclang's core is private and installed from a clone: the core's vector-store
record module names it and hands its record on, and every other module, of the
core, the toolkit, a use case or the charts package, imports the record from
there. Read from every distribution's ``src/`` with ``ast``, so a new
distribution is checked with no edit here; tests are left out, since the core's
build doclang's chunks by hand.
"""

from __future__ import annotations

import ast

from repo_paths import CORE_ROOT, REPO_ROOT

DOCLANG = "doclang_domain"
# The one module that may import it.
RECORD_MODULE = (
    CORE_ROOT / "src" / "query_processing_core" / "retrieval" / "vector_store_record.py"
)


def imports_doclang(source: str) -> bool:
    """Whether ``source`` imports doclang, by an absolute import of its namespace.

    Example::

        imports_doclang("from doclang_domain.core.chunking import Chunk")  # True
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module or ""]
        else:
            continue
        if any(module.split(".")[0] == DOCLANG for module in modules):
            return True
    return False


def test_the_doclang_reader_on_literal_sources() -> None:
    assert imports_doclang("from doclang_domain.core.chunking import Chunk")
    assert imports_doclang("import doclang_domain.core")
    assert not imports_doclang("from .doclang_domain import x")
    assert not imports_doclang("import doclang_domain_extra")
    assert not imports_doclang("from query_processing_core.retrieval import x")


def test_one_module_of_the_distributions_imports_doclang() -> None:
    modules = sorted(REPO_ROOT.glob("*/src/**/*.py"))
    importers = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in modules
        if imports_doclang(path.read_text(encoding="utf-8"))
    ]

    assert modules, f"no module was read under {REPO_ROOT}/*/src: the walk is wrong"
    assert importers == [RECORD_MODULE.relative_to(REPO_ROOT).as_posix()]
