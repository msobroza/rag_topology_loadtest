"""Rule 4 of the README's "Where a file goes": the adapters hold ports and their
implementations, nothing else.

Each check is a pure function shown failing on a literal input first: the
adapters hold only the port packages (each component's, the orchestrator's and
the key source's), each with its port in ``base_<name>.py`` and one directory
per registered name holding one implementation of the port, the ``api``
directory implementing both ``served`` and ``api``; no port is implemented
anywhere else. Moved out of ``test_placement.py``, which was at its 500 lines.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from query_processing.adapters.key_source.base_key_source import KEY_SOURCES
from query_processing.adapters.orchestrator.base_orchestrator import ORCHESTRATORS
from query_processing.adapters.router.fake.fake_router import FakeRouter
from query_processing.composition.openers.component_openers import (
    EMBEDDER_OPENERS,
    LLM_OPENERS,
    RERANKER_OPENERS,
    VECTOR_STORE_OPENERS,
)
from repo_paths import TOOLKIT_ROOT

PACKAGE = TOOLKIT_ROOT / "src" / "query_processing"

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
# Every package of adapters/: the components, the orchestrator, the library
# that runs the graph (ADR 0010), and the key source, where an API key comes
# from (ADR 0012), each laid out as a component is, with one directory per
# registered name.
PORT_PACKAGES = {
    **COMPONENTS,
    "orchestrator": ({"BaseOrchestrator"}, ORCHESTRATORS.names()),
    "key_source": ({"BaseKeySource"}, KEY_SOURCES.names()),
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
