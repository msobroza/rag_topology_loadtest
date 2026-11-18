"""The directory-size rule, measured on folders each test builds under ``tmp_path``."""

from __future__ import annotations

from pathlib import Path

from query_processing.testing import (
    FolderNotYetSplit,
    OnePatternFolder,
    directory_size_findings,
)

LIMIT = 3


def modules(folder: Path, *names: str) -> None:
    """Write an empty module of each name in ``folder``, with an ``__init__.py``."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in (*names, "__init__.py"):
        (folder / name).write_text("", encoding="utf-8")


def test_a_folder_at_the_limit_passes(tmp_path: Path) -> None:
    modules(tmp_path / "src" / "pkg", "a.py", "b.py", "c.py")

    assert directory_size_findings(tmp_path, limit=LIMIT) == []


def test_a_folder_over_the_limit_is_named_with_its_count(tmp_path: Path) -> None:
    modules(tmp_path / "tests" / "crowded", "a.py", "b.py", "c.py", "conftest.py")
    modules(tmp_path / "src" / "pkg", "a.py")

    (finding,) = directory_size_findings(tmp_path, limit=LIMIT)

    assert finding.startswith("tests/crowded: 4 modules, over the limit of 3;")
    assert "split it by what its modules share" in finding


CHARTS = OnePatternFolder(folder="src/charts", pattern=r"[a-z0-9_]+_chart\.py")


def test_a_one_pattern_folder_passes_whatever_its_length(tmp_path: Path) -> None:
    modules(tmp_path / "src" / "charts", *(f"c{n}_chart.py" for n in range(7)))

    findings = directory_size_findings(
        tmp_path, limit=LIMIT, one_pattern_folders=[CHARTS]
    )

    assert findings == []


def test_a_one_pattern_folder_names_the_module_off_its_pattern(tmp_path: Path) -> None:
    modules(tmp_path / "src" / "charts", "a_chart.py", "b_chart.py", "helpers.py")

    (finding,) = directory_size_findings(
        tmp_path, limit=LIMIT, one_pattern_folders=[CHARTS]
    )

    assert finding.startswith("src/charts/helpers.py: does not match")
    assert CHARTS.pattern in finding


def test_a_listed_one_pattern_folder_that_is_gone_is_named(tmp_path: Path) -> None:
    modules(tmp_path / "src" / "pkg", "a.py")

    (finding,) = directory_size_findings(
        tmp_path, limit=LIMIT, one_pattern_folders=[CHARTS]
    )

    assert finding.startswith("src/charts: listed as a one-pattern folder")


DOMAIN = FolderNotYetSplit(folder="src/domain", ceiling=5, reason="a later change")


def five_modules_in_domain(tmp_path: Path, *more: str) -> None:
    modules(tmp_path / "src" / "domain", "a.py", "b.py", "c.py", "d.py", "e.py", *more)


def test_a_folder_not_yet_split_passes_at_its_ceiling(tmp_path: Path) -> None:
    five_modules_in_domain(tmp_path)

    assert directory_size_findings(tmp_path, limit=LIMIT, not_yet_split=[DOMAIN]) == []


def test_a_folder_not_yet_split_fails_over_its_ceiling(tmp_path: Path) -> None:
    five_modules_in_domain(tmp_path, "f.py")

    (finding,) = directory_size_findings(tmp_path, limit=LIMIT, not_yet_split=[DOMAIN])

    assert finding.startswith("src/domain: 6 modules, over its ceiling of 5")
    assert "a later change" in finding


def test_a_folder_not_yet_split_is_stale_once_at_the_limit(tmp_path: Path) -> None:
    modules(tmp_path / "src" / "domain", "a.py", "b.py", "c.py")

    (finding,) = directory_size_findings(tmp_path, limit=LIMIT, not_yet_split=[DOMAIN])

    assert finding.startswith("src/domain: 3 modules, at or under the limit of 3;")
    assert "remove it from the folders not yet split" in finding


def test_a_folder_not_yet_split_that_is_gone_is_stale(tmp_path: Path) -> None:
    modules(tmp_path / "src" / "pkg", "a.py")

    (finding,) = directory_size_findings(tmp_path, limit=LIMIT, not_yet_split=[DOMAIN])

    assert finding.startswith("src/domain: 0 modules, at or under the limit of 3;")
