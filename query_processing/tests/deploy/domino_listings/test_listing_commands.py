"""Tests of the listing commands' look ``"list"``: its items, and how it runs.

On the fake Domino, whose environment env-1 has the revisions 7 (latest), 6
(active) and 5. No test runs the list itself: a named fake records what would
have been shown, and ``test_opening_list`` drives the list.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
import yaml

from fake_domino.server import create_fake_domino, fake_domino_http
from fake_domino.stand_ins import fake_environment, fake_tier
from fake_domino.state import FakeDominoApps
from query_processing.deploy.domino_listings.listing_commands import (
    environment_items,
    show_tiers,
    tier_items,
)
from query_processing.deploy.domino_listings.opening_list import OpeningItem
from query_processing.deploy.domino_listings.tiers_listing import (
    tiers_details,
    tiers_table,
)
from query_processing.deploy.domino_lookups.environments import list_environments
from query_processing.deploy.domino_lookups.hardware_tiers import list_hardware_tiers
from query_processing.deploy.domino_lookups.internal_revisions import (
    InternalRevisions,
)

TOOLKIT = Path(__file__).resolve().parents[3]
BUILT_7 = "2026-11-12T09:41:00Z"
BUILT_6 = "2026-11-03T16:02:00Z"
ENV_1 = fake_environment(
    "env-1",
    (7, "rev-7", BUILT_7),
    (6, "rev-6", BUILT_6),
    (5, "rev-5", BUILT_6),
    name="qp-rag",
    active=6,
)
FAKE_TIERS = [
    fake_tier("small-k8s", "cpu-8", 8, 32, is_default=True),
    fake_tier("large-k8s", "cpu-16", 16, 64),
    fake_tier("gpu-k8s", "gpu-1", 8, 32, gpus=1),
]
ROUTE_404 = (
    "the internal route GET /v4/environments/env-1/paginatedRevisionData returned "
    "HTTP 404, so the revisions it would list are not known"
)
NO_EXTRA = (
    'qp-deploy: the interactive list needs the toolkit\'s "terminal" extra '
    '(pip install -e "query_processing[terminal]" from the repository\'s root); '
    "here is the table\n"
)


class RecordedLists:
    """Stands in for ``run_opening_list``: records each list it was asked to run."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, list[OpeningItem]]] = []

    def __call__(self, title: str, items: Sequence[OpeningItem]) -> None:
        self.runs.append((title, list(items)))


def fake_apps() -> FakeDominoApps:
    return FakeDominoApps(tiers=FAKE_TIERS, environments=[ENV_1])


def http_on(apps: FakeDominoApps) -> httpx.Client:
    return fake_domino_http(create_fake_domino(apps))


def internal_calls(apps: FakeDominoApps) -> list[tuple[str, str]]:
    return [call for call in apps.received if call[1].endswith("RevisionData")]


def test_a_tiers_item_is_its_name_size_and_id_and_opens_on_every_field() -> None:
    tiers = list_hardware_tiers(http_on(fake_apps()))
    items = tier_items(tiers, opened=())
    assert items[1].label == "cpu-16   16 cores, 64 GiB   id large-k8s"
    assert [item.name for item in items] == ["cpu-8", "cpu-16", "gpu-1"]
    assert list(items[1].lines()) == tiers_details([tiers[1]]).splitlines()
    assert [item.starts_open for item in items] == [False, False, False]


def test_an_environment_asks_its_revisions_only_when_opened() -> None:
    apps = fake_apps()
    http = http_on(apps)
    [environment] = list_environments(http)
    [item] = environment_items([environment], InternalRevisions(http), opened=())
    assert item.label == "qp-rag   id env-1"
    assert item.name == "qp-rag"
    assert internal_calls(apps) == []
    lines = list(item.lines())
    assert lines[:4] == [
        "id env-1",
        "v7  latest  built 2026-11-12 09:41  id rev-7",
        "v6  active  built 2026-11-03 16:02  id rev-6",
        "v5          built 2026-11-03 16:02  id rev-5",
    ]
    [fields] = yaml.safe_load("\n".join(lines[4:]))
    assert fields["name"] == "qp-rag"
    assert fields["latestRevision"]["number"] == 7
    assert len(internal_calls(apps)) == 1
    item.lines()
    assert len(internal_calls(apps)) == 1  # not asked again


def test_an_environment_whose_route_fails_shows_the_three_and_the_note() -> None:
    apps = fake_apps()
    apps.revisions_route_status = 404
    http = http_on(apps)
    [item] = environment_items(list_environments(http), InternalRevisions(http), ())
    lines = list(item.lines())
    assert lines[1].startswith("v7  latest  built -")
    assert lines[2].startswith("v6  active  built -")
    assert ROUTE_404 in lines


def test_a_named_item_starts_open() -> None:
    tiers = list_hardware_tiers(http_on(fake_apps()))
    items = tier_items(tiers, opened=["cpu-16"])
    assert [item.starts_open for item in items] == [False, True, False]


def test_the_list_look_runs_the_list_when_textual_is_there(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorded = RecordedLists()
    show_tiers(http_on(fake_apps()), [], "list", run_list=recorded)
    [(title, items)] = recorded.runs
    assert title == "Hardware tiers"
    assert len(items) == 3
    assert capsys.readouterr() == ("", "")


def test_the_list_look_without_the_extra_prints_the_table_and_how_to_install_it(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A None in sys.modules is how Python marks a module that cannot be imported:
    # find_spec answers None for it, as it does where Textual is not installed.
    monkeypatch.setitem(sys.modules, "textual", None)
    recorded = RecordedLists()
    http = http_on(fake_apps())
    show_tiers(http, [], "list", run_list=recorded)
    assert recorded.runs == []
    assert capsys.readouterr() == (tiers_table(list_hardware_tiers(http)), NO_EXTRA)


def test_the_core_dependencies_do_not_name_textual() -> None:
    project = tomllib.loads((TOOLKIT / "pyproject.toml").read_text())["project"]
    assert not [d for d in project["dependencies"] if d.startswith("textual")]
    extras = project["optional-dependencies"]
    assert "textual>=8.2" in extras["terminal"]
    assert "textual>=8.2" in extras["dev"]


def test_only_the_opening_list_imports_textual() -> None:
    importing = sorted(
        path.name
        for path in (TOOLKIT / "src" / "query_processing").rglob("*.py")
        if "textual" in _imported(path)
    )
    assert importing == ["opening_list.py"]


def _imported(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    return names | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
