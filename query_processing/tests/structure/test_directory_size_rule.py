"""Rule 9 of the README's "Where a file goes" on the toolkit: at most 12 modules a
directory under ``src/`` and ``tests/``, ``__init__.py`` not counted.

The two lists are the rule's two exceptions. A one-pattern folder may be as
long as its list, and every module of it must match its pattern, so a mixed
folder cannot hide behind the exception. A folder not yet split may not pass
the count it had when listed, and its entry fails as stale once it is back at
12 or under, so the list only shrinks.
"""

from __future__ import annotations

from query_processing.testing import (
    FolderNotYetSplit,
    OnePatternFolder,
    directory_size_findings,
)
from repo_paths import TOOLKIT_ROOT

ONE_PATTERN_FOLDERS = (
    # One opener per component type and Backend; the pattern names no component
    # type, so a type a use case adds needs no edit here.
    OnePatternFolder(
        folder="src/query_processing/composition/openers",
        pattern=r"[a-z][a-z0-9_]*_on_[a-z][a-z0-9]*\.py|component_openers\.py",
    ),
    OnePatternFolder(
        folder="tests/entrypoints/qp_serve",
        pattern=r"test_qp_serve(_[a-z0-9_]+)?\.py",
    ),
)
NOT_YET_SPLIT = (
    FolderNotYetSplit(
        folder="tests/helpers",
        ceiling=21,
        reason="imported by plain module name in a hundred test files",
    ),
    FolderNotYetSplit(
        folder="src/fake_domino/emulation",
        ceiling=16,
        reason="temporary: deleted with the emulated Domino",
    ),
    FolderNotYetSplit(
        folder="tests/fake_domino",
        ceiling=15,
        reason="ten of its modules test the emulated Domino and go with it",
    ),
)


def test_no_directory_of_the_toolkit_holds_more_than_twelve_modules() -> None:
    findings = directory_size_findings(
        TOOLKIT_ROOT,
        one_pattern_folders=ONE_PATTERN_FOLDERS,
        not_yet_split=NOT_YET_SPLIT,
    )
    assert not findings, "\n".join(findings)
