"""The README's rules that apply to the core, measured on the core.

The core (``query_processing_core``, ADR 0014) is a distribution of its own, and
its tests may not import the toolkit, so the toolkit's structure tests read its
files as they read the repository's: rule 3 (no two modules share a file name,
the core's and the toolkit's together), rule 9 (twelve modules a directory), the
README's tree listing its modules and nothing else, and the code shape of
``test_code_shape.py``. Its manifest names doclang's core by name and version
alone (the spec's story 42): no URL and no path, since each install hands the
package over from the clone or from its wheel.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path

from test_code_shape import (
    MAX_BODY_LINES,
    MAX_FILE_LINES,
    MAX_NESTING,
    body_lines,
    functions,
    nesting,
)
from test_placement import (
    PACKAGE,
    readme_tree,
    shared_file_names,
    tree_mismatches,
    tree_root_folder,
)

from query_processing.testing import directory_size_findings
from repo_paths import CORE_ROOT, REPO_ROOT

CORE_PACKAGE = CORE_ROOT / "src" / "query_processing_core"
CORE_TREE_ROOT = "src/query_processing_core/"  # where the README's tree of it starts
DOCLANG_CORE = "doclang-domain-core"
# A name and version specifiers alone: ">=0.1,<0.2", "==0.1.3".
BY_NAME_AND_VERSION = re.compile(
    rf"{DOCLANG_CORE}\s*(?:[<>=!~]=?\s*[0-9][0-9A-Za-z.*+-]*\s*,?\s*)+"
)
# What locates a package rather than naming it: a URL, a file or a path.
LOCATION = re.compile(r"://|\bfile:|@|(?:^|\s)\.{0,2}/|\\")


def core_python_files() -> list[Path]:
    """Every ``.py`` file of the core, its source and its tests."""
    return sorted((CORE_ROOT / "src").rglob("*.py")) + sorted(
        (CORE_ROOT / "tests").rglob("*.py")
    )


def test_no_module_of_the_core_shares_a_file_name_with_another() -> None:
    core = [path.relative_to(REPO_ROOT) for path in CORE_PACKAGE.rglob("*.py")]
    toolkit = [path.relative_to(REPO_ROOT) for path in PACKAGE.rglob("*.py")]

    assert core, f"no module found under {CORE_PACKAGE}"
    assert shared_file_names(core + toolkit) == {}


def test_no_directory_of_the_core_holds_more_than_twelve_modules() -> None:
    assert (CORE_ROOT / "src").is_dir()
    assert (CORE_ROOT / "tests").is_dir()
    assert directory_size_findings(CORE_ROOT) == []


def test_the_readme_tree_lists_every_module_of_the_core_and_nothing_else() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert tree_root_folder(readme, CORE_TREE_ROOT) == f"{CORE_ROOT.name}/"
    assert tree_mismatches(readme_tree(readme, CORE_TREE_ROOT), CORE_PACKAGE) == []


def measured_core() -> Iterator[tuple[str, int, int]]:
    """``(file:line name, body lines, nesting)`` for each function of the core."""
    for path in sorted(CORE_PACKAGE.rglob("*.py")):
        for function in functions(path.read_text(encoding="utf-8")):
            where = f"{path.relative_to(CORE_ROOT)}:{function.lineno} {function.name}"
            yield where, body_lines(function), nesting(function)


def test_the_core_keeps_the_code_shape() -> None:
    too_long = [
        f"{path.relative_to(CORE_ROOT)}: {count} lines"
        for path in core_python_files()
        if (count := len(path.read_text(encoding="utf-8").splitlines()))
        > MAX_FILE_LINES
    ]
    measured = list(measured_core())
    long_bodies = [f"{w}: {n} lines" for w, n, _ in measured if n > MAX_BODY_LINES]
    too_deep = [f"{w}: {d} levels" for w, _, d in measured if d > MAX_NESTING]

    assert measured, f"no function found under {CORE_PACKAGE}: the walk is wrong"
    assert (too_long, long_bodies, too_deep) == ([], [], [])


def located_requirements(manifest: dict[str, object]) -> list[str]:
    """What ``manifest``, a parsed ``pyproject.toml``, says of doclang's core
    beyond its name and version, one sentence each.

    Example::

        located_requirements({"project": {"dependencies": ["doclang-domain-core>=0.1"]}})
        # []
    """
    project = manifest.get("project", {})
    assert isinstance(project, dict), f"[project] is not a table: {project!r}"
    named = [d for d in project.get("dependencies", []) if DOCLANG_CORE in d]
    if not named:
        return [f"{DOCLANG_CORE} is not a dependency"]
    found = [
        f"{dependency!r}: not a name and version alone"
        for dependency in named
        if not BY_NAME_AND_VERSION.fullmatch(dependency.strip())
    ]
    located = [text for text in strings_of(manifest) if LOCATION.search(text)]
    return found + [f"{text!r}: locates a package" for text in located]


def strings_of(value: object) -> Iterator[str]:
    """Every string of a parsed TOML ``value`` that names doclang's core."""
    if isinstance(value, dict):
        for item in value.values():
            yield from strings_of(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_of(item)
    elif isinstance(value, str) and "doclang" in value.lower():
        yield value


def test_the_requirement_reader_on_literal_manifests() -> None:
    def project(*dependencies: str) -> dict[str, object]:
        return {"project": {"dependencies": list(dependencies)}}

    assert located_requirements(project("doclang-domain-core>=0.1,<0.2")) == []
    assert located_requirements(project("doclang-domain-core==0.1.3")) == []
    assert located_requirements(project("pydantic>=2")) == [
        "doclang-domain-core is not a dependency"
    ]
    assert located_requirements(project("doclang-domain-core")) == [
        "'doclang-domain-core': not a name and version alone"
    ]
    assert located_requirements(
        project("doclang-domain-core @ file:///home/me/document_doclang/core")
    ) == [
        "'doclang-domain-core @ file:///home/me/document_doclang/core': not a "
        "name and version alone",
        "'doclang-domain-core @ file:///home/me/document_doclang/core': locates "
        "a package",
    ]
    sources = {"tool": {"uv": {"sources": {"doclang": {"path": "../doclang/core"}}}}}
    assert located_requirements({**project("doclang-domain-core>=0.1"), **sources}) == [
        "'../doclang/core': locates a package"
    ]


def test_the_core_names_doclang_s_core_by_name_and_version_alone() -> None:
    manifest = tomllib.loads((CORE_ROOT / "pyproject.toml").read_text("utf-8"))

    assert located_requirements(manifest) == []


# What the core holds, as each place that says it words it; the history is part
# of a request, so a place may name it or not.
WHAT_THE_CORE_HOLDS = ("metadata filter", "request", "response", "vector-store record")


def between(text: str, begin: str, end: str) -> str:
    """The part of ``text`` from ``begin`` up to the next ``end``, both required.

    Example::

        between("a - x - b", "- x", "- b")   # "- x "
    """
    assert begin in text, f"no {begin!r} in the text"
    start = text.index(begin)
    assert end in text[start:], f"no {end!r} after {begin!r} in the text"
    return text[start : text.index(end, start)]


def places_that_say_what_the_core_holds() -> dict[str, str]:
    """Each place that says what the core holds, by name, with its words."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    glossary = (REPO_ROOT / "CONTEXT.md").read_text(encoding="utf-8")
    manifest = tomllib.loads((CORE_ROOT / "pyproject.toml").read_text("utf-8"))
    package = (CORE_PACKAGE / "__init__.py").read_text(encoding="utf-8")
    return {
        "README.md": between(readme, "- **The core**", "- **The toolkit**"),
        "CLAUDE.md": between(
            claude, "- `query_processing_core/`", "- `query_processing/`"
        ),
        "CONTEXT.md": between(glossary, "**Core**:", "_Avoid_"),
        "pyproject.toml": manifest["project"]["description"],
        "__init__.py": ast.get_docstring(ast.parse(package)) or "",
    }


def test_each_place_says_the_same_of_what_the_core_holds() -> None:
    # A phrase may wrap at a line's end.
    said = {
        place: " ".join(text.split()).lower()
        for place, text in places_that_say_what_the_core_holds().items()
    }
    missing = {
        place: [thing for thing in WHAT_THE_CORE_HOLDS if thing not in text]
        for place, text in said.items()
    }

    assert {place: things for place, things in missing.items() if things} == {}
