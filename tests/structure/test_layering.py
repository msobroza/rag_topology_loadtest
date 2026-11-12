"""The import rule, read from the source with ``ast``: what each of the seven
packages of ``rag_load_test`` may import (the table in the README's "Where a file
goes"). And the mock rule: ``fake_domino`` imports nothing from ``rag_load_test``,
and only ``deploy/rag_deploy.py`` (``--dry-run``) imports ``fake_domino``; the
mock's own tests, in ``tests/fake_domino/``, import nothing from ``rag_load_test``
either."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from repo_paths import REPO_ROOT

PACKAGE = REPO_ROOT / "src" / "rag_load_test"
FAKE_DOMINO = PACKAGE.parent / "fake_domino"
FAKE_DOMINO_IMPORTERS = {Path("deploy") / "rag_deploy.py"}  # relative to PACKAGE
# Package -> the packages of rag_load_test its modules may import.
MAY_IMPORT: dict[str, set[str]] = {
    "domain": {"domain"},
    "application": {"application", "domain"},
    "config": {"config", "domain"},
    "adapters": {"adapters", "domain"},
    "entrypoints": {"adapters", "application", "config", "domain"},
    "deploy": {"deploy", "domain"},
    "loadtest": {"loadtest", "domain"},
}
# A port: adapters/<component>/base_<component>.py. The application depends on
# the ports and on nothing else of the adapters; a port imports only the domain.
PORT_MODULE = re.compile(r"rag_load_test\.adapters\.(\w+)\.base_\1")
MAY_IMPORT_PORTS = {"application"}
# The named exceptions: one module each, allowed more than its package.
ALSO_MAY_IMPORT: dict[str, set[str]] = {
    "rag_load_test.adapters.rag_ports": {"config", "application"},  # builds the ports
    # For the defaults of the four RAG_* values a load test sends, nothing else.
    "rag_load_test.loadtest.loadtest_settings": {"config"},
}
LOCUSTFILE = REPO_ROOT / "locustfile.py"  # the load-test client's file, outside src


def module_name(path: Path) -> str:
    """The dotted name of the module at ``path``, a file under ``src/``.

    Example::

        module_name(PACKAGE / "domain" / "passages.py")   # "rag_load_test.domain.passages"
    """
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imported_product_modules(module: str, source: str) -> list[tuple[int, str]]:
    """``(line, module)`` for every import of a module of ``rag_load_test``.

    Relative imports are resolved against ``module``, at any depth.

    Example::

        imported_product_modules("rag_load_test.adapters.llm.api.api_llm",
                                 "from ....domain.passages import Passage")
        # [(1, "rag_load_test.domain.passages")]
    """
    home = module.split(".")[:-1]  # the package a relative import starts from
    found = []
    for node in ast.walk(ast.parse(source)):
        for imported in _imported_modules(node, home):
            parts = imported.split(".")
            if parts[0] == "rag_load_test" and len(parts) > 1:
                found.append((node.lineno, imported))
    return found


def imported_packages(module: str, source: str) -> list[tuple[int, str]]:
    """``(line, package)`` for every import of a package of ``rag_load_test``.

    Example::

        imported_packages("rag_load_test.adapters.llm.api.api_llm",
                          "from ....domain.passages import Passage")   # [(1, "domain")]
    """
    return [
        (line, imported.split(".")[1])
        for line, imported in imported_product_modules(module, source)
    ]


def _imported_modules(node: ast.AST, home: list[str]) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if node.level == 0:
        return [node.module or ""]
    base = home[: len(home) - node.level + 1]
    if node.module:  # ``from ..x import y`` imports x
        return [".".join([*base, node.module])]
    return [
        ".".join([*base, alias.name]) for alias in node.names
    ]  # ``from .. import x``


def forbidden_imports(module: str, source: str) -> list[str]:
    """What ``module`` imports from a package the table does not allow it.

    Example::

        forbidden_imports("rag_load_test.domain.passages",
                          "from ..adapters import rag_ports")
        # ["rag_load_test.domain.passages:1 imports rag_load_test.adapters"]
    """
    package = module.split(".")[1]
    allowed = MAY_IMPORT[package] | ALSO_MAY_IMPORT.get(module, set())
    may_import_ports = package in MAY_IMPORT_PORTS
    return [
        f"{module}:{line} imports {imported}"
        for line, imported in imported_product_modules(module, source)
        if imported.split(".")[1] not in allowed
        and not (may_import_ports and PORT_MODULE.fullmatch(imported))
    ]


def test_the_import_reader_on_literal_sources() -> None:
    nested = "rag_load_test.adapters.llm.api.api_llm"
    assert imported_packages(nested, "from ....domain.passages import x") == [
        (1, "domain")
    ]
    assert imported_packages(nested, "from ...http.json_post import x") == [
        (1, "adapters")
    ]
    assert imported_packages(nested, "from .... import config") == [(1, "config")]
    assert imported_packages(nested, "import rag_load_test.deploy.status") == [
        (1, "deploy")
    ]
    assert imported_packages(nested, "import httpx\nfrom http import HTTPStatus") == []


def test_the_import_table_on_literal_sources() -> None:
    assert forbidden_imports(
        "rag_load_test.domain.passages", "from ..adapters import rag_ports"
    ) == ["rag_load_test.domain.passages:1 imports rag_load_test.adapters"]
    assert forbidden_imports(
        "rag_load_test.loadtest.request_plans", "\nfrom ..config import rag_settings"
    ) == ["rag_load_test.loadtest.request_plans:2 imports rag_load_test.config"]
    port = "from ..adapters.embedder.base_embedder import BaseEmbedder"
    assert forbidden_imports("rag_load_test.application.workflow", port) == []
    assert forbidden_imports("rag_load_test.config.rag_settings", port) != []
    fake = "from ..adapters.embedder.fake.fake_embedder import FakeEmbedder"
    assert forbidden_imports("rag_load_test.application.workflow", fake) == [
        "rag_load_test.application.workflow:1 imports "
        "rag_load_test.adapters.embedder.fake.fake_embedder"
    ]
    ports = "from ..config.rag_settings import RagSettings"
    assert forbidden_imports("rag_load_test.adapters.rag_ports", ports) == []
    assert forbidden_imports("rag_load_test.adapters.json_log", ports) != []
    settings = "from ..config.rag_settings import RagSettings"
    assert forbidden_imports("rag_load_test.loadtest.loadtest_settings", settings) == []


def test_the_table_names_every_package_of_the_product() -> None:
    packages = {path.parent.name for path in PACKAGE.glob("*/__init__.py")}
    assert packages == set(MAY_IMPORT)


@pytest.mark.parametrize("package", sorted(MAY_IMPORT))
def test_each_package_imports_only_what_the_table_allows(package: str) -> None:
    offending = [
        offence
        for path in sorted((PACKAGE / package).rglob("*.py"))
        for offence in forbidden_imports(
            module_name(path), path.read_text(encoding="utf-8")
        )
    ]
    allowed = ", ".join(sorted(MAY_IMPORT[package]))
    assert not offending, f"{package} may import only {allowed}: {offending}"


def test_the_ports_import_only_the_domain() -> None:
    ports = [
        path
        for path in sorted((PACKAGE / "adapters").glob("*/base_*.py"))
        if PORT_MODULE.fullmatch(module_name(path))
    ]
    offending = [
        f"{module_name(path)}:{line} imports {imported}"
        for path in ports
        for line, imported in imported_product_modules(
            module_name(path), path.read_text(encoding="utf-8")
        )
        if imported.split(".")[1] != "domain"
    ]
    assert len(ports) == 4, f"expected the four ports, found {ports}"
    assert not offending, f"a port may import only the domain: {offending}"


def test_the_locustfile_reads_only_the_load_test_client() -> None:
    imported = absolute_imports(LOCUSTFILE, "rag_load_test")
    offending = [
        f"locustfile.py:{line} imports {module}"
        for line, module in imported
        if not module.startswith("rag_load_test.loadtest.")
    ]
    assert imported, "the locustfile imports nothing from rag_load_test: wrong file?"
    assert (
        not offending
    ), f"the locustfile may import only the loadtest package: {offending}"


def absolute_imports(path: Path, package: str) -> list[tuple[int, str]]:
    """``(line, module)`` for every absolute import of ``package`` in ``path``.

    A relative import cannot name another top-level package, so these are all.

    Example::

        absolute_imports(PACKAGE / "deploy" / "rag_deploy.py", "fake_domino")
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


def test_the_fake_domino_tests_import_nothing_from_the_product() -> None:
    tests = REPO_ROOT / "tests" / "fake_domino"
    offending = [
        f"{path.relative_to(tests.parent)}:{line} imports {module}"
        for path in sorted(tests.rglob("*.py"))
        for line, module in absolute_imports(path, "rag_load_test")
    ]
    assert tests.is_dir(), f"{tests} is where the fake Domino's own tests live"
    assert not offending, f"tests/fake_domino may not import rag_load_test: {offending}"


def test_only_rag_deploy_imports_fake_domino() -> None:
    offending = [
        f"{path.relative_to(PACKAGE)}:{line} imports {module}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.relative_to(PACKAGE) not in FAKE_DOMINO_IMPORTERS
        for line, module in absolute_imports(path, "fake_domino")
    ]
    assert (
        not offending
    ), f"only deploy/rag_deploy.py may import fake_domino: {offending}"
