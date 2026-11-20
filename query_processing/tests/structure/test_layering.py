"""The import rule, read from the source with ``ast``: what each of the ten
packages of ``query_processing`` may import (the table in the README's "Where a file
goes"). And the mock rule: ``fake_domino`` imports nothing from ``query_processing``,
and only ``deploy/qp_deploy.py`` (``--dry-run``) imports ``fake_domino``; the
mock's own tests, in ``tests/fake_domino/``, import nothing from ``query_processing``
either. The one-way rule, that no file of the toolkit's source names a use case, is
in test_one_way_to_use_cases.py."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from repo_paths import TOOLKIT_ROOT

PACKAGE = TOOLKIT_ROOT / "src" / "query_processing"
FAKE_DOMINO = PACKAGE.parent / "fake_domino"
FAKE_DOMINO_IMPORTERS = {Path("deploy") / "qp_deploy.py"}  # relative to PACKAGE
# What a use case imports of the toolkit: a module, not a package, outside the
# table below. It may import every package, and none imports it.
EXTENSION_API = PACKAGE / "extension_api.py"
# Package -> the packages of query_processing its modules may import.
MAY_IMPORT: dict[str, set[str]] = {
    "domain": {"domain"},
    "application": {"application", "domain"},
    "config": {"config", "domain"},
    "adapters": {"adapters", "infrastructure", "domain"},
    "infrastructure": {"infrastructure", "domain"},
    # The composition root: the one package that builds the adapters, so the one
    # that may import both them and the settings.
    "composition": {
        "composition",
        "adapters",
        "application",
        "config",
        "domain",
        "infrastructure",
    },
    # A command gets its ports from the composition root, never from an adapter.
    "entrypoints": {
        "composition",
        "application",
        "config",
        "domain",
        "infrastructure",
    },
    "deploy": {"deploy", "domain"},
    "loadtest": {"loadtest", "domain"},
    # The tests the toolkit ships (``ComponentContract``, ``RouterContract``):
    # they serve a user's component as the toolkit does, so they may import
    # every package, and no package imports them, extension_api included: a use
    # case's tests import them from query_processing.testing.
    "testing": {
        "testing",
        "domain",
        "application",
        "config",
        "adapters",
        "infrastructure",
        "composition",
        "entrypoints",
        "deploy",
        "loadtest",
    },
}
# A port: adapters/<component>/base_<component>.py, which imports only the domain.
# The application imports ports and nothing else of the adapters.
PORT_MODULE = re.compile(r"query_processing\.adapters\.(\w+)\.base_\1")
MAY_IMPORT_PORTS = {"application"}
# The named exceptions: one module each, and what it may import beyond its
# package's table: a package, or a module, each with what lies under it.
KEY_SOURCE_PORT = "query_processing.adapters.key_source.base_key_source"
STANDALONE_KEY_SOURCE = "query_processing.composition.standalone_key_source"
ALSO_MAY_IMPORT: dict[str, set[str]] = {
    # The defaults of a load test's QP_* values and retries; its key and domino blocks.
    "query_processing.loadtest.loadtest_settings": {"query_processing.config"},
    # The key source of the load test's key block, which composition opens, and
    # the port, to name its type.
    "query_processing.loadtest.target_client": {STANDALONE_KEY_SOURCE, KEY_SOURCE_PORT},
    # The deploy tool reads the workflow settings (ADR 0003), checks the overrides,
    # asks the composition's two start-up checks what an App would refuse, and
    # its key_source_variables what the key sources read; the rest of deploy
    # reads them through it.
    "query_processing.deploy.workflow_components": {
        "query_processing.config",
        "query_processing.composition.composition_root",
        "query_processing.composition.registered_names",
    },
    # qp-deploy status sends the run token, which only a key source asks for.
    "query_processing.deploy.status": {STANDALONE_KEY_SOURCE, KEY_SOURCE_PORT},
}
# The named exceptions that load modules by name, as an extension is loaded,
# from a package their own may not import: one module each, and the modules
# it may name. Nothing else names a module of the toolkit in a string.
LOADED_BY_NAME: dict[str, set[str]] = {
    # The settings read the components of the toolkit's own types, so they load
    # the five ports and the router's package (its fake backend, its API
    # format) first; config never imports them (TOOLKIT_COMPONENT_TYPES).
    "query_processing.config.extensions": {
        "query_processing.adapters.embedder.base_embedder",
        "query_processing.adapters.reranker.base_reranker",
        "query_processing.adapters.llm.base_llm",
        "query_processing.adapters.vector_store.base_vector_store",
        "query_processing.adapters.router",
    },
}
# A string that is the dotted name of a module of the toolkit.
DOTTED_MODULE_NAME = re.compile(r"query_processing(\.\w+)+")
LOCUSTFILE = TOOLKIT_ROOT / "locustfile.py"  # the load-test client's file, outside src


def module_name(path: Path) -> str:
    """The dotted name of the module at ``path``, a file under ``src/``.

    Example::

        module_name(PACKAGE / "domain" / "registry.py")   # "query_processing.domain.registry"
    """
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def importer_name(path: Path) -> str:
    """The name the relative imports of the file at ``path`` resolve against.

    An ``__init__.py`` resolves them from its own package, so its name keeps
    ``__init__`` as the last part, the one ``imported_product_modules`` drops.

    Example::

        importer_name(PACKAGE / "adapters" / "llm" / "api" / "__init__.py")
        # "query_processing.adapters.llm.api.__init__"
    """
    return ".".join(path.relative_to(PACKAGE.parent).with_suffix("").parts)


def imported_product_modules(module: str, source: str) -> list[tuple[int, str]]:
    """``(line, module)`` for every import of a module of ``query_processing``.

    Relative imports are resolved against ``module``, at any depth.

    Example::

        imported_product_modules("query_processing.adapters.llm.api.api_llm",
                                 "from ....domain.registry import Registry")
        # [(1, "query_processing.domain.registry")]
    """
    home = module.split(".")[:-1]  # the package a relative import starts from
    found = []
    for node in ast.walk(ast.parse(source)):
        for imported in _imported_modules(node, home):
            parts = imported.split(".")
            if parts[0] == "query_processing" and len(parts) > 1:
                found.append((node.lineno, imported))
    return found


def imported_packages(module: str, source: str) -> list[tuple[int, str]]:
    """``(line, package)`` for every import of a package of ``query_processing``.

    Example::

        imported_packages("query_processing.adapters.llm.api.api_llm",
                          "from ....domain.registry import Registry")   # [(1, "domain")]
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

        forbidden_imports("query_processing.domain.registry",
                          "from ..adapters import embedder")
        # ["query_processing.domain.registry:1 imports query_processing.adapters"]
    """
    may_import_ports = module.split(".")[1] in MAY_IMPORT_PORTS
    return [
        f"{module}:{line} imports {imported}"
        for line, imported in imported_product_modules(module, source)
        if not table_allows(module, imported)
        and not (may_import_ports and PORT_MODULE.fullmatch(imported))
    ]


def table_allows(module: str, imported: str) -> bool:
    """Whether the table lets ``module`` import, or load, the module ``imported``.

    Its package's row, or what ``ALSO_MAY_IMPORT`` names for it alone, a module
    named there with what lies under it.

    Example::

        table_allows("query_processing.deploy.status",
                     "query_processing.composition.composition_root")   # False
    """
    if imported.split(".")[1] in MAY_IMPORT[module.split(".")[1]]:
        return True
    return any(
        imported == named or imported.startswith(f"{named}.")
        for named in ALSO_MAY_IMPORT.get(module, set())
    )


def test_the_import_reader_on_literal_sources() -> None:
    nested = "query_processing.adapters.llm.api.api_llm"
    assert imported_packages(nested, "from ....domain.registry import x") == [
        (1, "domain")
    ]
    assert imported_packages(nested, "from ...http.json_post import x") == [
        (1, "adapters")
    ]
    assert imported_packages(nested, "from .... import config") == [(1, "config")]
    assert imported_packages(nested, "import query_processing.deploy.status") == [
        (1, "deploy")
    ]
    assert imported_packages(nested, "import httpx\nfrom http import HTTPStatus") == []
    package_init = importer_name(PACKAGE / "adapters" / "llm" / "api" / "__init__.py")
    assert package_init == "query_processing.adapters.llm.api.__init__"
    assert imported_packages(package_init, "from ....domain.registry import x") == [
        (1, "domain")
    ]
    assert forbidden_imports(
        package_init, "from ....config import workflow_settings"
    ) == [f"{package_init}:1 imports query_processing.config"]


def test_the_import_table_on_literal_sources() -> None:
    assert forbidden_imports(
        "query_processing.domain.registry", "from ..adapters import embedder"
    ) == ["query_processing.domain.registry:1 imports query_processing.adapters"]
    assert forbidden_imports(
        "query_processing.loadtest.request_plans",
        "\nfrom ..config import workflow_settings",
    ) == ["query_processing.loadtest.request_plans:2 imports query_processing.config"]
    port = "from ..adapters.embedder.base_embedder import BaseEmbedder"
    assert forbidden_imports("query_processing.application.workflow", port) == []
    assert forbidden_imports("query_processing.config.workflow_settings", port) != []
    fake = "from ..adapters.embedder.fake.fake_embedder import FakeEmbedder"
    assert forbidden_imports("query_processing.application.workflow", fake) == [
        "query_processing.application.workflow:1 imports "
        "query_processing.adapters.embedder.fake.fake_embedder"
    ]
    ports = "from ..config.workflow_settings import WorkflowSettings"
    root = "query_processing.composition.composition_root"
    assert forbidden_imports(root, ports) == []
    assert forbidden_imports("query_processing.infrastructure.json_log", ports) != []
    absolute = "from query_processing.config.workflow_settings import WorkflowSettings"
    port_module = "query_processing.adapters.embedder.base_embedder"
    assert forbidden_imports(port_module, absolute) != []
    adapter = "from ..adapters.embedder.fake.fake_embedder import FakeEmbedder"
    assert forbidden_imports(root, adapter) == []
    assert forbidden_imports("query_processing.entrypoints.qp_serve", adapter) != []
    settings = "from ..config.workflow_settings import WorkflowSettings"
    assert (
        forbidden_imports("query_processing.loadtest.loadtest_settings", settings) == []
    )
    workflow = "from ..config.files.workflow_file import workflow_file"
    assert (
        forbidden_imports("query_processing.deploy.workflow_components", workflow) == []
    )
    assert forbidden_imports("query_processing.deploy.publish", workflow) == [
        "query_processing.deploy.publish:1 imports query_processing.config.files.workflow_file"
    ]


def test_the_table_names_every_package_of_the_product() -> None:
    packages = {path.parent.name for path in PACKAGE.glob("*/__init__.py")}
    assert packages == set(MAY_IMPORT)


@pytest.mark.parametrize("package", sorted(MAY_IMPORT))
def test_each_package_imports_only_what_the_table_allows(package: str) -> None:
    offending = [
        offence
        for path in sorted((PACKAGE / package).rglob("*.py"))
        for offence in forbidden_imports(
            importer_name(path), path.read_text(encoding="utf-8")
        )
    ]
    allowed = ", ".join(sorted(MAY_IMPORT[package]))
    assert not offending, f"{package} may import only {allowed}: {offending}"


def forbidden_names(module: str, source: str) -> list[str]:
    """The modules ``module`` names in a string, to load them, beyond what it may.

    A module may name what its package may import, and what ``LOADED_BY_NAME``
    gives it.

    Example::

        forbidden_names("query_processing.domain.registry",
                        "X = 'query_processing.adapters.router'")
        # ["query_processing.domain.registry:1 names query_processing.adapters.router"]
    """
    named = LOADED_BY_NAME.get(module, set())
    return [
        f"{module}:{node.lineno} names {node.value}"
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and DOTTED_MODULE_NAME.fullmatch(node.value)
        and not table_allows(module, node.value)
        and node.value not in named
    ]


def test_the_name_reader_on_literal_sources() -> None:
    router = "X = 'query_processing.adapters.router'"
    assert forbidden_names("query_processing.domain.registry", router) == [
        "query_processing.domain.registry:1 names query_processing.adapters.router"
    ]
    assert forbidden_names("query_processing.config.extensions", router) == []
    fake = "X = 'query_processing.adapters.router.fake.fake_router'"
    assert forbidden_names("query_processing.config.extensions", fake) != []
    prose = '"""Loads query_processing.adapters.router first."""'
    assert forbidden_names("query_processing.domain.registry", prose) == []
    assert (
        forbidden_names("query_processing.config.x", "X = 'query_processing.domain'")
        == []
    )


@pytest.mark.parametrize("package", sorted(MAY_IMPORT))
def test_each_package_loads_by_name_only_what_the_table_allows(package: str) -> None:
    offending = [
        offence
        for path in sorted((PACKAGE / package).rglob("*.py"))
        for offence in forbidden_names(
            module_name(path), path.read_text(encoding="utf-8")
        )
    ]
    assert not offending, f"{package} names modules it may not load: {offending}"


def test_the_settings_load_the_modules_the_table_gives_them() -> None:
    from query_processing.config.extensions import TOOLKIT_COMPONENT_TYPES

    named = LOADED_BY_NAME["query_processing.config.extensions"]
    assert set(TOOLKIT_COMPONENT_TYPES) == named


def importers_of(top: str, exempt: Path | None = None) -> list[str]:
    """Each import of ``query_processing.<top>`` from outside it, ``exempt`` aside.

    Example::

        importers_of("testing")   # [] when the rule holds
    """
    return [
        f"{module_name(path)}:{line} imports {imported}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path != exempt and PACKAGE / top not in path.parents
        for line, imported in imported_product_modules(
            importer_name(path), path.read_text(encoding="utf-8")
        )
        if imported.split(".")[1] == top
    ]


def test_no_package_of_the_toolkit_imports_the_extension_api() -> None:
    # extension_api may import every package: it is what a use case imports,
    # and the toolkit never depends on what it offers a use case.
    offending = importers_of(EXTENSION_API.stem, EXTENSION_API)
    assert EXTENSION_API.is_file(), f"{EXTENSION_API} is what a use case imports"
    assert not offending, f"no package may import extension_api: {offending}"


def test_no_package_of_the_toolkit_imports_the_tests_it_ships() -> None:
    # The toolkit never runs them, and extension_api, which every command
    # imports with a use case's extensions, would load the model server and
    # the test client with them.
    offending = importers_of("testing")
    assert (PACKAGE / "testing" / "__init__.py").is_file()
    assert not offending, f"no module may import testing: {offending}"


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
    # The five components' ports, the orchestrator's (ADR 0010) and the key
    # source's (ADR 0012).
    assert len(ports) == 7, f"expected the seven ports, found {ports}"
    assert not offending, f"a port may import only the domain: {offending}"


def test_the_locustfile_reads_only_the_load_test_client() -> None:
    imported = absolute_imports(LOCUSTFILE, "query_processing")
    offending = [
        f"locustfile.py:{line} imports {module}"
        for line, module in imported
        if not module.startswith("query_processing.loadtest.")
    ]
    assert imported, "the locustfile imports nothing from query_processing: wrong file?"
    assert (
        not offending
    ), f"the locustfile may import only the loadtest package: {offending}"


def absolute_imports(path: Path, package: str) -> list[tuple[int, str]]:
    """``(line, module)`` for every absolute import of ``package`` in ``path``.

    A relative import cannot name another top-level package, so these are all.

    Example::

        absolute_imports(PACKAGE / "deploy" / "qp_deploy.py", "fake_domino")
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
        for line, module in absolute_imports(path, "query_processing")
    ]
    assert not offending, f"fake_domino may not import query_processing: {offending}"


def test_the_fake_domino_tests_import_nothing_from_the_product() -> None:
    tests = TOOLKIT_ROOT / "tests" / "fake_domino"
    offending = [
        f"{path.relative_to(tests.parent)}:{line} imports {module}"
        for path in sorted(tests.rglob("*.py"))
        for line, module in absolute_imports(path, "query_processing")
    ]
    assert tests.is_dir(), f"{tests} is where the fake Domino's own tests live"
    assert (
        not offending
    ), f"tests/fake_domino may not import query_processing: {offending}"


def test_only_qp_deploy_imports_fake_domino() -> None:
    offending = [
        f"{path.relative_to(PACKAGE)}:{line} imports {module}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.relative_to(PACKAGE) not in FAKE_DOMINO_IMPORTERS
        for line, module in absolute_imports(path, "fake_domino")
    ]
    assert (
        not offending
    ), f"only deploy/qp_deploy.py may import fake_domino: {offending}"
