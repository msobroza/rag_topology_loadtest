"""The emulation is temporary: it must stay deletable without breaking the rest.

Its code, settings and documentation live in the places listed under "Removing
the emulation" in ``emulation/README.md``; nothing else may come to depend on it,
and the project's own settings files may not mention it. This file goes away
with the emulation too.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

from repo_paths import REPO_ROOT

EMULATION_PACKAGE = "fake_domino.emulation"
EMULATION_COMMAND = "rag-domino-emulation"
# Where the emulation lives, and the few shared files that name it (each of them
# is on the removal list).
EMULATION_HOMES = (
    "emulation/",
    "tests/emulation/",
    "src/fake_domino/emulation/",
    "tests/fake_domino/test_fake_domino_emulation_",
)
SHARED_FILES_THAT_NAME_IT = {
    "pyproject.toml",  # the script entry, the extra, the test dependencies
    "README.md",  # one pointer to emulation/README.md
    "CONTEXT.md",  # the "Emulation" block of the glossary
    "docs/adr/0001-emulation-platform-and-app-placement.md",
    ".gitignore",
}
# The project's own settings: what the owner reads to configure a real Domino.
PROJECT_SETTINGS_FILES = (
    "src/rag_load_test/config/rag_settings.py",
    "src/rag_load_test/config/domino_rules.py",
    "src/rag_load_test/deploy/deploy_settings.py",
    "deploy.example.yaml",
    ".env.example",
    "Makefile",
)


def tracked_files() -> list[str]:
    """Every file git knows or would add (tracked, plus untracked and not ignored)."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return [name for name in listed.stdout.splitlines() if (REPO_ROOT / name).is_file()]


def is_emulation_home(name: str) -> bool:
    return name.startswith(EMULATION_HOMES)


def text_of(name: str) -> str:
    try:
        return (REPO_ROOT / name).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""  # a binary file names nothing


def imported_modules(path: Path) -> set[str]:
    """Absolute module names a Python file imports, relative ones resolved."""
    package = ".".join(path.relative_to(REPO_ROOT / "src").with_suffix("").parts[:-1])
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package.split(".")[: len(package.split(".")) - node.level + 1]
            prefix = ".".join(base) if node.level else ""
            module = ".".join(part for part in (prefix, node.module or "") if part)
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def test_only_the_listed_places_name_the_emulation() -> None:
    naming = [
        name
        for name in tracked_files()
        if not is_emulation_home(name)
        and (EMULATION_PACKAGE in text_of(name) or EMULATION_COMMAND in text_of(name))
    ]

    assert set(naming) <= SHARED_FILES_THAT_NAME_IT, (
        "these files name the emulation but are not on its removal list "
        f"(emulation/README.md): {sorted(set(naming) - SHARED_FILES_THAT_NAME_IT)}"
    )


@pytest.mark.parametrize("name", PROJECT_SETTINGS_FILES)
def test_the_projects_own_settings_do_not_mention_the_emulation(name: str) -> None:
    mentions = re.findall(r"(?i)emulat\w*", text_of(name))

    assert mentions == [], f"{name} mentions the emulation: {mentions}"


def test_nothing_outside_the_emulation_imports_it() -> None:
    sources = [
        path
        for path in sorted((REPO_ROOT / "src").rglob("*.py"))
        if not is_emulation_home(str(path.relative_to(REPO_ROOT)))
    ]
    offending = [
        str(path.relative_to(REPO_ROOT))
        for path in sources
        if any(m.startswith(EMULATION_PACKAGE) for m in imported_modules(path))
    ]

    assert sources, "no source file was read: the walk is wrong"
    assert not offending, f"only the emulation may import itself: {offending}"


def test_the_removal_list_names_every_home_of_the_emulation() -> None:
    readme = text_of("emulation/README.md")
    removal = readme[readme.index("Removing the emulation") :]
    expected = (
        "emulation/",
        "tests/emulation/",
        "src/fake_domino/emulation/",
        "test_fake_domino_emulation_",
        EMULATION_COMMAND,
        "pyproject.toml",
        "CONTEXT.md",
        "README.md",
    )

    assert [item for item in expected if item not in removal] == []
