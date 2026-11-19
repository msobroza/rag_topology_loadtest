"""Where a file goes and what it is called (the README's "Where a file goes").

Four checks, each a pure function shown failing on a literal input first:
every command of the product lives in a module named after it, in its
application's package; the README's module tree lists every package and module
of ``query_processing`` and nothing else; no two modules share a file name; the
adapters hold only the component packages, each with its port in
``base_<component>.py`` and one directory per Backend holding one implementation
of the port, the ``api`` directory implementing both ``served`` and ``api``, and
no port is implemented anywhere else. The
fake Domino is not the product: its commands and modules are outside the rule.
"""

from __future__ import annotations

import ast
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from query_processing.adapters.orchestrator.base_orchestrator import ORCHESTRATORS
from query_processing.adapters.router.fake.fake_router import FakeRouter
from query_processing.composition.openers.component_openers import (
    EMBEDDER_OPENERS,
    LLM_OPENERS,
    RERANKER_OPENERS,
    VECTOR_STORE_OPENERS,
)
from repo_paths import REPO_ROOT, TOOLKIT_ROOT

PACKAGE = TOOLKIT_ROOT / "src" / "query_processing"
# The two tools have their commands in their own package; every other command
# of the product belongs to the RAG system, whose commands are in entrypoints.
COMMAND_HOME = {
    "qp-deploy": "deploy",
    "qp-compare": "loadtest",
    "qp-record-answers": "loadtest",
    "qp-compare-answers": "loadtest",
}
RAG_SYSTEM_COMMAND_HOME = "entrypoints"
TREE_ROOT = "src/query_processing/"  # where the README's tree of the product starts
TREE_INDENT = 2  # spaces per level in the tree
TOOLKIT_FOLDER = f"{TOOLKIT_ROOT.name}/"  # what the README's tree nests it in


# --- commands ---------------------------------------------------------------------


def misplaced_commands(scripts: dict[str, str], package: Path) -> list[str]:
    """The commands of ``scripts`` that are not where the rule puts them.

    Example::

        misplaced_commands({"qp-serve": "query_processing.entrypoints.app:main"}, PACKAGE)
        # ["qp-serve -> query_processing.entrypoints.app, expected ...qp_serve"]
    """
    wrong = []
    for command, target in sorted(scripts.items()):
        module = target.split(":")[0]
        if module.startswith("fake_domino."):
            continue  # the fake Domino is outside the product
        home = COMMAND_HOME.get(command, RAG_SYSTEM_COMMAND_HOME)
        expected = f"query_processing.{home}.{command.replace('-', '_')}"
        if module != expected:
            wrong.append(f"{command} -> {module}, expected {expected}")
        elif not (package / home / f"{command.replace('-', '_')}.py").is_file():
            wrong.append(f"{command} -> {module}, which has no file")
    return wrong


def test_the_command_rule_on_literal_scripts() -> None:
    scripts = {
        "qp-serve": "query_processing.entrypoints.workflow_api:main",  # wrong name
        "qp-deploy": "query_processing.entrypoints.qp_deploy:main",  # wrong package
        "qp-ingest": "query_processing.entrypoints.qp_ingest:main",
        "qp-gone": "query_processing.entrypoints.qp_gone:main",  # no such file
        "qp-fake-domino": "fake_domino.server:main",
    }
    assert misplaced_commands(scripts, PACKAGE) == [
        "qp-deploy -> query_processing.entrypoints.qp_deploy, "
        "expected query_processing.deploy.qp_deploy",
        "qp-gone -> query_processing.entrypoints.qp_gone, which has no file",
        "qp-serve -> query_processing.entrypoints.workflow_api, "
        "expected query_processing.entrypoints.qp_serve",
    ]


def test_every_command_lives_in_a_module_named_after_it() -> None:
    pyproject = tomllib.loads((TOOLKIT_ROOT / "pyproject.toml").read_text("utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert any(target.startswith("query_processing.") for target in scripts.values())
    assert misplaced_commands(scripts, PACKAGE) == []


# --- the README's module tree -------------------------------------------------------


def readme_tree(readme: str) -> set[str]:
    """The paths the README's tree lists under ``src/query_processing/``.

    A package is listed as ``name/``, a file by its name; each level is indented
    two spaces more than its parent. Lines that name no file or folder (wrapped
    descriptions) are skipped.

    Example::

        readme_tree("```text\\nsrc/query_processing/\\n  domain/\\n    ports.py  x\\n```")
        # {"domain/", "domain/ports.py"}
    """
    listed: set[str] = set()
    folders: list[str] = []
    for indent, name in _tree_entries(readme):
        depth = indent // TREE_INDENT
        where = f"README tree: {name!r} is indented {indent} spaces under {folders}"
        assert indent % TREE_INDENT == 0, where
        assert depth <= len(folders) + 1, where
        del folders[depth - 1 :]
        listed.add("".join(folders) + name)
        if name.endswith("/"):
            folders.append(name)
    return listed


def _tree_entries(readme: str) -> Iterable[tuple[int, str]]:
    # (indent, name) for each line of the product's tree that names a path, the
    # indent counted from the tree root's own: the root may sit in a folder.
    lines = readme.splitlines()
    start = _tree_start(lines)
    base = _indent(lines[start])
    for line in lines[start + 1 :]:
        indent = _indent(line) - base
        if indent <= 0:  # the next entry at the root's level or above, or the end
            return
        name = line.split()[0]
        if name.endswith(("/", ".py", ".yaml")):
            yield indent, name


def _tree_start(lines: list[str]) -> int:
    starts = [i for i, line in enumerate(lines) if line.split()[:1] == [TREE_ROOT]]
    assert starts, f"README: no tree starting with a {TREE_ROOT!r} line"
    return starts[0]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def tree_root_folder(readme: str) -> str:
    """The folder the README's tree nests ``src/query_processing/`` in, "" for none.

    Example::

        tree_root_folder("```text\\nqp/\\n  src/query_processing/\\n```")   # "qp/"
    """
    lines = readme.splitlines()
    start = _tree_start(lines)
    base = _indent(lines[start])
    if base == 0:
        return ""
    above = [line for line in lines[:start] if line.strip() and _indent(line) < base]
    return above[-1].split()[0]


def product_tree(package: Path) -> set[str]:
    """Every package (``name/``) and module of ``package``, ``__init__.py`` aside.

    Example::

        product_tree(PACKAGE)   # {"domain/", "domain/ports.py", ...}
    """
    packages = {
        f"{path.parent.relative_to(package).as_posix()}/"
        for path in package.rglob("__init__.py")
        if path.parent != package
    }
    modules = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if path.name != "__init__.py"
    }
    return packages | modules


def tree_mismatches(listed: set[str], package: Path) -> list[str]:
    """What the listed tree misses, and what it names that does not exist.

    Example::

        tree_mismatches({"domain/", "domain/gone.py"}, PACKAGE)
        # ["not in the README: domain/ports.py", ..., "not in src: domain/gone.py"]
    """
    present = product_tree(package)
    code = {path for path in listed if path.endswith(("/", ".py"))}
    other_files = {path for path in listed - code if path != "__init__.py"}
    missing = sorted(present - code - {"__init__.py"})
    absent = sorted(path for path in code - {"__init__.py"} if path not in present)
    absent += sorted(path for path in other_files if not (package / path).exists())
    return [f"not in the README: {p}" for p in missing] + [
        f"not in src: {p}" for p in absent
    ]


LITERAL_README = """## Code layout

```text
src/query_processing/
  __init__.py       the package
  domain/           no I/O
    ports.py        the ports, described on
                    two lines (this one names no file)
    gone.py         a module that was deleted
  config/
    prompts.yaml    a data file
src/fake_domino/    another package: not read
  server.py
```
"""


LITERAL_NESTED_README = """```text
toolkit/            the distribution
  src/query_processing/
    domain/
      ports.py      the ports
  src/fake_domino/  another package: not read
    server.py
docs/               beside the toolkit: not read
```
"""


def test_the_tree_reader_on_a_literal_readme() -> None:
    assert readme_tree(LITERAL_README) == {
        "__init__.py",
        "domain/",
        "domain/ports.py",
        "domain/gone.py",
        "config/",
        "config/prompts.yaml",
    }
    assert tree_root_folder(LITERAL_README) == ""


def test_the_tree_reader_on_a_tree_inside_a_folder() -> None:
    assert readme_tree(LITERAL_NESTED_README) == {"domain/", "domain/ports.py"}
    assert tree_root_folder(LITERAL_NESTED_README) == "toolkit/"


def test_the_tree_comparison_on_a_literal_package(tmp_path: Path) -> None:
    for name in ("__init__.py", "domain/__init__.py", "domain/ports.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "domain" / "rules.py").write_text("", encoding="utf-8")

    assert tree_mismatches(readme_tree(LITERAL_README), tmp_path) == [
        "not in the README: domain/rules.py",
        "not in src: config/",
        "not in src: domain/gone.py",
        "not in src: config/prompts.yaml",
    ]


def test_the_readme_tree_lists_every_module_and_nothing_else() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert tree_root_folder(readme) == TOOLKIT_FOLDER
    assert tree_mismatches(readme_tree(readme), PACKAGE) == []


# --- one name, one module -----------------------------------------------------------


def shared_file_names(paths: Iterable[Path]) -> dict[str, list[str]]:
    """File names that more than one of ``paths`` has, ``__init__.py`` aside.

    Example::

        shared_file_names([Path("a/x.py"), Path("b/x.py")])   # {"x.py": [...]}
    """
    by_name: defaultdict[str, list[str]] = defaultdict(list)
    for path in paths:
        if path.name != "__init__.py":
            by_name[path.name].append(path.as_posix())
    return {name: sorted(found) for name, found in by_name.items() if len(found) > 1}


def test_the_name_rule_on_literal_paths() -> None:
    paths = [Path("a/x.py"), Path("b/x.py"), Path("c/y.py"), Path("a/__init__.py")]
    paths.append(Path("b/__init__.py"))

    assert shared_file_names(paths) == {"x.py": ["a/x.py", "b/x.py"]}


def test_no_two_modules_of_the_product_share_a_file_name() -> None:
    modules = [path.relative_to(PACKAGE) for path in PACKAGE.rglob("*.py")]

    assert modules, f"no module found under {PACKAGE}"
    assert shared_file_names(modules) == {}


# --- adapters: only ports and their implementations, one package per component ------

ADAPTERS = PACKAGE / "adapters"
# A Backend implemented in another Backend's directory: served and api differ
# only in how the endpoint is resolved, so the one HTTP implementation, in api/,
# serves both (the owner's decision: two directories would hold one class twice).
SHARED_DIRECTORIES = {"served": "api"}
# Component package -> the ports it holds besides its own, each in its own
# base_*.py: a store's filter translator turns a metadata filter into the store's
# own filter, one implementation in each Backend directory beside the store.
OTHER_PORTS: dict[str, set[str]] = {
    "vector_store": {"base_metadata_filter_translator.py"},
}
# Component package -> the API format written by hand beside its port: the
# router's, the worked example of one, for its served backend, which is built
# and so has no directory.
HAND_WRITTEN_FORMATS: dict[str, set[str]] = {"router": {"router_api_format.py"}}
# Component package -> its port classes, and the Backend values that name its
# implementation directories: those registered for each component type. The
# router's are those its classes name, fake; its served class is built.
COMPONENTS: dict[str, tuple[set[str], tuple[str, ...]]] = {
    "embedder": ({"BaseEmbedder"}, EMBEDDER_OPENERS.names()),
    "reranker": ({"BaseReranker"}, RERANKER_OPENERS.names()),
    "llm": ({"BaseLLM"}, LLM_OPENERS.names()),
    "vector_store": (
        {"BaseVectorStore", "BaseLexicalSearch"},
        VECTOR_STORE_OPENERS.names(),
    ),
    "router": ({"BaseRouter"}, tuple(FakeRouter.backends)),
}
# Every package of adapters/: the components, and the orchestrator, the library
# that runs the graph (ADR 0010), laid out as a component is, with one
# directory per registered name.
PORT_PACKAGES = {
    **COMPONENTS,
    "orchestrator": ({"BaseOrchestrator"}, ORCHESTRATORS.names()),
}


def port_implementations(source: str, ports: set[str]) -> list[str]:
    """Names of the classes in ``source`` that subclass one of ``ports``.

    Example::

        port_implementations("class A(BaseEmbedder): ...", {"BaseEmbedder"})  # ["A"]
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {
            b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
            for b in node.bases
        }
        if bases & ports:
            found.append(node.name)
    return found


def component_problems(
    component: Path, ports: set[str], backends: tuple[str, ...]
) -> list[str]:
    """What breaks the layout of one component package.

    The layout: ``base_<component>.py`` holds the port (``OTHER_PORTS`` names
    any other port beside it, ``HAND_WRITTEN_FORMATS`` an API format written
    by hand), and each Backend value is
    a directory holding exactly one implementation of it, except a Backend of
    ``SHARED_DIRECTORIES``, implemented in the directory it names.

    Example::

        component_problems(ADAPTERS / "embedder", {"BaseEmbedder"}, ("local", "api"))
        # [] when embedder/ has base_embedder.py, local/ and api/, one class each
    """
    name = component.name
    shared = {b: SHARED_DIRECTORIES[b] for b in backends if b in SHARED_DIRECTORIES}
    backends = tuple(sorted({SHARED_DIRECTORIES.get(b, b) for b in backends}))
    problems = (
        []
        if (component / f"base_{name}.py").is_file()
        else [f"{name}: no base_{name}.py"]
    )
    expected_files = {
        "__init__.py",
        f"base_{name}.py",
        *OTHER_PORTS.get(name, ()),
        *HAND_WRITTEN_FORMATS.get(name, ()),
    }
    problems += [
        f"{name}/{path.name} is neither the port nor a Backend directory"
        for path in sorted(component.glob("*.py"))
        if path.name not in expected_files
    ]
    directories = {path.parent.name for path in component.glob("*/__init__.py")}
    problems += [f"{name}: no {b}/ directory" for b in backends if b not in directories]
    problems += [
        (
            f"{name}/{d}/ is not a Backend directory: {d} is implemented in {shared[d]}/"
            if d in shared
            else f"{name}/{d}/ is not a Backend"
        )
        for d in sorted(directories - set(backends))
    ]
    for backend in sorted(directories & set(backends)):
        classes = [
            cls
            for path in sorted((component / backend).rglob("*.py"))
            for cls in port_implementations(path.read_text(encoding="utf-8"), ports)
        ]
        if len(classes) != 1:
            problems.append(
                f"{name}/{backend}/ implements the port {len(classes)} times: {classes}"
            )
    return problems


def test_the_component_layout_on_a_literal_package(tmp_path: Path) -> None:
    sources = {
        "base_embedder.py": "class BaseEmbedder: ...",
        "local/__init__.py": "",
        "local/st.py": "class StEmbedder(BaseEmbedder): ...",
        "api/__init__.py": "",
        "api/a.py": "class A(BaseEmbedder): ...\nclass B(base_embedder.BaseEmbedder): ...",
        "gpu/__init__.py": "",
        "served/__init__.py": "",
        "served/s.py": "class S(BaseEmbedder): ...",
        "http_helpers.py": "def get(): ...",
    }
    for name, source in sources.items():
        (tmp_path / "embedder" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "embedder" / name).write_text(source, encoding="utf-8")

    assert component_problems(
        tmp_path / "embedder", {"BaseEmbedder"}, ("local", "served", "api", "fake")
    ) == [
        "embedder/http_helpers.py is neither the port nor a Backend directory",
        "embedder: no fake/ directory",
        "embedder/gpu/ is not a Backend",
        "embedder/served/ is not a Backend directory: served is implemented in api/",
        "embedder/api/ implements the port 2 times: ['A', 'B']",
    ]
    assert component_problems(tmp_path / "reranker", {"BaseReranker"}, ()) == [
        "reranker: no base_reranker.py"
    ]


def test_each_component_has_its_port_and_one_directory_per_backend() -> None:
    problems = [
        problem
        for name, (ports, backends) in PORT_PACKAGES.items()
        for problem in component_problems(ADAPTERS / name, ports, backends)
    ]
    assert problems == []


def test_every_port_implementation_lives_in_its_component() -> None:
    misplaced = [
        f"{path.relative_to(PACKAGE)}: {cls}"
        for name, (ports, _) in PORT_PACKAGES.items()
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.relative_to(PACKAGE).parts[:2] != ("adapters", name)
        for cls in port_implementations(path.read_text(encoding="utf-8"), ports)
    ]
    assert misplaced == []


def not_adapters(adapters: Path, components: Iterable[str]) -> list[str]:
    """What sits directly in ``adapters/`` and is not a component package.

    Only ports and their implementations are adapters: code that implements no
    port belongs to ``infrastructure/``, and what builds the adapters from the
    settings to ``composition/``.

    Example::

        not_adapters(ADAPTERS, COMPONENTS)   # [] when only the components are there
    """
    allowed = {*components, "__init__.py", "__pycache__"}
    return sorted(path.name for path in adapters.iterdir() if path.name not in allowed)


def test_the_adapters_rule_on_a_literal_package(tmp_path: Path) -> None:
    for name in ("embedder", "http", "reranker"):
        (tmp_path / name).mkdir()
    for name in ("__init__.py", "json_log.py", "rag_ports.py"):
        (tmp_path / name).write_text("", encoding="utf-8")

    assert not_adapters(tmp_path, ["embedder", "reranker"]) == [
        "http",
        "json_log.py",
        "rag_ports.py",
    ]


def test_adapters_holds_only_the_component_packages() -> None:
    assert not_adapters(ADAPTERS, PORT_PACKAGES) == []
