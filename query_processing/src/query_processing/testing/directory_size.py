"""The directory-size rule of the README's "Where a file goes", for any distribution.

A directory holds at most ``limit`` modules: the ``.py`` files directly inside
it, ``__init__.py`` not counted and ``conftest.py`` counted. Every directory
under the distribution's ``src/`` and ``tests/`` is measured. Two kinds of
folder are told apart by the caller, never by this module, which knows no
folder of its own: a one-pattern folder, whose every module is one entry of
one sort and may be as long as its list, and a folder not yet split, which
may not pass its ceiling and is stale once back at the limit.

Example:
    >>> from pathlib import Path
    >>> directory_size_findings(Path("query_processing"))  # doctest: +SKIP
    []
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# The two folders of a distribution the rule measures, each with its tree.
MEASURED_TREES = ("src", "tests")
NOT_A_MODULE = "__init__.py"
# The number of rule 9, written once: the default of ``limit``, which the
# toolkit's own test and a use case's leave alone to follow the README.
MAX_MODULES_PER_DIRECTORY = 12


@dataclass(frozen=True)
class OnePatternFolder:
    """A folder that may be as long as its list, since every module is one entry.

    Attributes:
        folder: its path from the distribution's root, with ``/``
            (``"src/pkg/openers"``).
        pattern: the regular expression every module's file name matches whole.
    """

    folder: str
    pattern: str


@dataclass(frozen=True)
class FolderNotYetSplit:
    """A crowded folder whose split is a later change: it may not grow.

    Attributes:
        folder: its path from the distribution's root, with ``/``.
        ceiling: the count it may not pass, its count when it was listed.
        reason: why it is not split yet, said in the finding.
    """

    folder: str
    ceiling: int
    reason: str


def directory_size_findings(
    root: Path,
    *,
    limit: int = MAX_MODULES_PER_DIRECTORY,
    one_pattern_folders: Sequence[OnePatternFolder] = (),
    not_yet_split: Sequence[FolderNotYetSplit] = (),
) -> list[str]:
    """What breaks the directory-size rule under ``root``, one sentence each.

    Args:
        root: the distribution's root, which holds its ``src/`` and ``tests/``.
        limit: the most modules a directory may hold; the README's 12 unless
            given.
        one_pattern_folders: the folders exempt from the limit by their pattern.
        not_yet_split: the folders over the limit, each held to its ceiling.

    Returns:
        Each finding, naming the folder, its count and what to do; empty when
        the rule holds.

    Example:
        >>> directory_size_findings(Path("."))  # doctest: +SKIP
        []
    """
    folders = dict(_module_lists(root))
    patterns = {listed.folder: listed.pattern for listed in one_pattern_folders}
    held_folders = {listed.folder for listed in not_yet_split}
    findings = [
        _gone_one_pattern_folder(name) for name in patterns if name not in folders
    ]
    for listed in not_yet_split:
        findings += _held_to_its_ceiling(
            listed, len(folders.get(listed.folder, [])), limit
        )
    for folder, names in folders.items():
        if folder in patterns:
            findings += _off_pattern(folder, names, patterns[folder])
        elif folder not in held_folders and len(names) > limit:
            findings.append(_over_the_limit(folder, len(names), limit))
    return findings


def _module_lists(root: Path) -> Iterator[tuple[str, list[str]]]:
    # ``(folder, its modules)`` for every directory under ``root``'s measured trees.
    for tree in MEASURED_TREES:
        top = root / tree
        directories = [top, *top.rglob("*")] if top.is_dir() else []
        for directory in sorted(d for d in directories if d.is_dir()):
            yield directory.relative_to(root).as_posix(), _modules_in(directory)


def _modules_in(directory: Path) -> list[str]:
    # The file names of the modules directly inside ``directory``.
    return sorted(p.name for p in directory.glob("*.py") if p.name != NOT_A_MODULE)


def _over_the_limit(folder: str, count: int, limit: int) -> str:
    # The finding for a folder over the limit that no exception covers.
    return (
        f"{folder}: {count} modules, over the limit of {limit}; split it by what "
        "its modules share (a component type, then a Backend, a command or a "
        "glossary area) into sub-packages named after that thing"
    )


def _off_pattern(folder: str, names: list[str], pattern: str) -> list[str]:
    # A finding for each module of a one-pattern folder its pattern does not match.
    return [
        f"{folder}/{name}: does not match {pattern}, the one pattern of its folder; "
        "rename it to match, or move it to a folder of its own sort"
        for name in names
        if not re.fullmatch(pattern, name)
    ]


def _gone_one_pattern_folder(folder: str) -> str:
    # The finding for a one-pattern folder the list names and the tree lacks.
    return (
        f"{folder}: listed as a one-pattern folder, but no such directory; "
        "remove it from the list"
    )


def _held_to_its_ceiling(
    listed: FolderNotYetSplit, count: int, limit: int
) -> list[str]:
    # The finding for a folder not yet split that grew, or that needs no listing.
    if count <= limit:
        return [
            f"{listed.folder}: {count} modules, at or under the limit of {limit}; "
            "remove it from the folders not yet split"
        ]
    if count > listed.ceiling:
        return [
            f"{listed.folder}: {count} modules, over its ceiling of {listed.ceiling} "
            f"as a folder not yet split ({listed.reason}); put the new module "
            "elsewhere, or split the folder"
        ]
    return []
