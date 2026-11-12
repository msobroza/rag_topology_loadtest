"""Where a file goes and what it is called (the README's "Where a file goes").

Four checks, each a pure function shown failing on a literal input first:
every command of the product lives in a module named after it, in its
application's package; the README's module tree lists every package and module
of ``rag_load_test`` and nothing else; no two modules share a file name; each
component of the adapters has its port in ``base_<component>.py`` and one
directory per Backend, each holding one implementation of the port. The
fake Domino is not the product: its commands and modules are outside the rule.
"""

from __future__ import annotations

import ast
import tomllib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import get_args

from rag_load_test.domain.backends import (
    EmbeddingBackend,
    GenerationBackend,
    RerankBackend,
    VectorStoreBackend,
)
from repo_paths import REPO_ROOT

PACKAGE = REPO_ROOT / "src" / "rag_load_test"
# The two tools have one command each, in their own package; every other command
# of the product belongs to the RAG system, whose commands are in entrypoints.
COMMAND_HOME = {"rag-deploy": "deploy", "rag-compare": "loadtest"}
RAG_SYSTEM_COMMAND_HOME = "entrypoints"
TREE_ROOT = "src/rag_load_test/"  # where the README's tree of the product starts
TREE_INDENT = 2  # spaces per level in the tree


# --- commands ---------------------------------------------------------------------


def misplaced_commands(scripts: dict[str, str], package: Path) -> list[str]:
    """The commands of ``scripts`` that are not where the rule puts them.

    Example::

        misplaced_commands({"rag-serve": "rag_load_test.entrypoints.app:main"}, PACKAGE)
        # ["rag-serve -> rag_load_test.entrypoints.app, expected ...rag_serve"]
    """
    wrong = []
    for command, target in sorted(scripts.items()):
        module = target.split(":")[0]
        if module.startswith("fake_domino."):
            continue  # the fake Domino is outside the product
        home = COMMAND_HOME.get(command, RAG_SYSTEM_COMMAND_HOME)
        expected = f"rag_load_test.{home}.{command.replace('-', '_')}"
        if module != expected:
            wrong.append(f"{command} -> {module}, expected {expected}")
        elif not (package / home / f"{command.replace('-', '_')}.py").is_file():
            wrong.append(f"{command} -> {module}, which has no file")
    return wrong


def test_the_command_rule_on_literal_scripts() -> None:
    scripts = {
        "rag-serve": "rag_load_test.entrypoints.workflow_api:main",  # wrong name
        "rag-deploy": "rag_load_test.entrypoints.rag_deploy:main",  # wrong package
        "rag-ingest": "rag_load_test.entrypoints.rag_ingest:main",
        "rag-gone": "rag_load_test.entrypoints.rag_gone:main",  # no such file
        "rag-fake-domino": "fake_domino.server:main",
    }
    assert misplaced_commands(scripts, PACKAGE) == [
        "rag-deploy -> rag_load_test.entrypoints.rag_deploy, "
        "expected rag_load_test.deploy.rag_deploy",
        "rag-gone -> rag_load_test.entrypoints.rag_gone, which has no file",
        "rag-serve -> rag_load_test.entrypoints.workflow_api, "
        "expected rag_load_test.entrypoints.rag_serve",
    ]


def test_every_command_lives_in_a_module_named_after_it() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert any(target.startswith("rag_load_test.") for target in scripts.values())
    assert misplaced_commands(scripts, PACKAGE) == []


# --- the README's module tree -------------------------------------------------------


def readme_tree(readme: str) -> set[str]:
    """The paths the README's tree lists under ``src/rag_load_test/``.

    A package is listed as ``name/``, a file by its name; each level is indented
    two spaces more than its parent. Lines that name no file or folder (wrapped
    descriptions) are skipped.

    Example::

        readme_tree("```text\\nsrc/rag_load_test/\\n  domain/\\n    ports.py  x\\n```")
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
    # (indent, name) for each line of the product's tree that names a path.
    lines = readme.splitlines()
    starts = [i for i, line in enumerate(lines) if line.split()[:1] == [TREE_ROOT]]
    assert starts, f"README: no tree starting with a {TREE_ROOT!r} line"
    for line in lines[starts[0] + 1 :]:
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:  # the next top-level entry, or the end of the block
            return
        name = line.split()[0]
        if name.endswith(("/", ".py", ".yaml")):
            yield indent, name


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
src/rag_load_test/
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


def test_the_tree_reader_on_a_literal_readme() -> None:
    assert readme_tree(LITERAL_README) == {
        "__init__.py",
        "domain/",
        "domain/ports.py",
        "domain/gone.py",
        "config/",
        "config/prompts.yaml",
    }


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
    listed = readme_tree((REPO_ROOT / "README.md").read_text(encoding="utf-8"))

    assert tree_mismatches(listed, PACKAGE) == []


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


# --- adapters: one package per component, one directory per Backend ----------------

ADAPTERS = PACKAGE / "adapters"
# Component package -> its port classes, and the Backend values that name its
# implementation directories.
COMPONENTS: dict[str, tuple[set[str], tuple[str, ...]]] = {
    "embedder": ({"BaseEmbedder"}, get_args(EmbeddingBackend)),
    "reranker": ({"BaseReranker"}, get_args(RerankBackend)),
    "llm": ({"BaseLLM"}, get_args(GenerationBackend)),
    "vector_store": (
        {"BaseVectorStore", "BaseLexicalSearch"},
        get_args(VectorStoreBackend),
    ),
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

    The layout: ``base_<component>.py`` holds the port, and each Backend value is
    a directory holding exactly one implementation of it.

    Example::

        component_problems(ADAPTERS / "embedder", {"BaseEmbedder"}, ("local", "api"))
        # [] when embedder/ has base_embedder.py, local/ and api/, one class each
    """
    name = component.name
    problems = (
        []
        if (component / f"base_{name}.py").is_file()
        else [f"{name}: no base_{name}.py"]
    )
    directories = {path.parent.name for path in component.glob("*/__init__.py")}
    problems += [f"{name}: no {b}/ directory" for b in backends if b not in directories]
    problems += [
        f"{name}/{d}/ is not a Backend" for d in sorted(directories - set(backends))
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
    }
    for name, source in sources.items():
        (tmp_path / "embedder" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "embedder" / name).write_text(source, encoding="utf-8")

    assert component_problems(
        tmp_path / "embedder", {"BaseEmbedder"}, ("local", "api", "fake")
    ) == [
        "embedder: no fake/ directory",
        "embedder/gpu/ is not a Backend",
        "embedder/api/ implements the port 2 times: ['A', 'B']",
    ]
    assert component_problems(tmp_path / "reranker", {"BaseReranker"}, ()) == [
        "reranker: no base_reranker.py"
    ]


def test_each_component_has_its_port_and_one_directory_per_backend() -> None:
    problems = [
        problem
        for name, (ports, backends) in COMPONENTS.items()
        for problem in component_problems(ADAPTERS / name, ports, backends)
    ]
    assert problems == []


def test_every_port_implementation_lives_in_its_component() -> None:
    misplaced = [
        f"{path.relative_to(ADAPTERS)}: {cls}"
        for name, (ports, _) in COMPONENTS.items()
        for path in sorted(ADAPTERS.rglob("*.py"))
        if path.relative_to(ADAPTERS).parts[0] != name
        for cls in port_implementations(path.read_text(encoding="utf-8"), ports)
    ]
    assert misplaced == []
