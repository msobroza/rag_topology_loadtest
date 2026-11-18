"""No directory of the charts package holds more than 12 modules.

Rule 9 of the README's "Where a file goes": the ``.py`` files directly inside a
directory under ``src/`` and ``tests/``, ``__init__.py`` not counted. ``charts/``
is the package's one-pattern folder: one module per finding, each named
``<finding>_chart.py``, so it may be as long as its list.

The toolkit ships the same measure (``query_processing.testing``), but this
package imports nothing of the toolkit (``test_experiment_charts_imports.py``),
so it measures with the few lines below instead of sharing that function.
"""

from __future__ import annotations

import re
from pathlib import Path

from experiment_chart_support import CHARTS_ROOT

MAX_MODULES = 12
ONE_PATTERN_FOLDERS = {
    "src/experiment_charts/charts": r"[a-z0-9_]+_chart\.py",
}


def modules_by_folder(root: Path) -> dict[str, list[str]]:
    """The modules directly inside each directory under ``root``'s src/ and tests/."""
    directories = [d for tree in ("src", "tests") for d in (root / tree).rglob("*")]
    return {
        d.relative_to(root).as_posix(): sorted(
            p.name for p in d.glob("*.py") if p.name != "__init__.py"
        )
        for d in [root / "src", root / "tests", *directories]
        if d.is_dir()
    }


def test_no_directory_holds_more_than_twelve_modules() -> None:
    crowded = {
        folder: len(names)
        for folder, names in modules_by_folder(CHARTS_ROOT).items()
        if folder not in ONE_PATTERN_FOLDERS and len(names) > MAX_MODULES
    }
    assert (
        not crowded
    ), f"over {MAX_MODULES} modules, split by what they share: {crowded}"


def test_every_module_of_a_one_pattern_folder_matches_its_pattern() -> None:
    folders = modules_by_folder(CHARTS_ROOT)
    gone = [folder for folder in ONE_PATTERN_FOLDERS if folder not in folders]
    assert not gone, f"listed as one-pattern folders, but no such directory: {gone}"
    off_pattern = [
        f"{folder}/{name}"
        for folder, pattern in ONE_PATTERN_FOLDERS.items()
        for name in folders[folder]
        if not re.fullmatch(pattern, name)
    ]
    assert not off_pattern, f"named off their folder's pattern: {off_pattern}"
