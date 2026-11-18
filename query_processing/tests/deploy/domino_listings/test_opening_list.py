"""Tests of the list that opens, driven by Textual's test pilot, never by a person.

Every test runs the list through ``run_test()``, which ends it when the test's
block ends; none calls ``App.run()``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from textual.widgets import Static, Tree

from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.domino_listings.opening_list import (
    OpeningItem,
    OpeningList,
)


def items(calls: list[str]) -> list[OpeningItem]:
    def lines_of(name: str) -> Callable[[], Sequence[str]]:
        def lines() -> Sequence[str]:
            calls.append(name)
            return [f"{name} line 1", f"{name} line 2"]

        return lines

    return [OpeningItem(name, lines_of(name)) for name in ("cpu-8", "cpu-16", "gpu-1")]


def shown(app: OpeningList) -> list[tuple[str, list[str]]]:
    """Each visible item with the lines under it when it is open."""
    tree = app.query_one(Tree)
    return [
        (
            str(node.label),
            [str(child.label) for child in node.children] if node.is_expanded else [],
        )
        for node in tree.root.children
    ]


async def test_space_opens_the_item_under_the_cursor_and_closes_it_again() -> None:
    calls: list[str] = []
    app = OpeningList("Hardware tiers", items(calls))
    async with app.run_test() as pilot:
        await pilot.press("down", "space")
        assert shown(app)[1] == ("cpu-16", ["cpu-16 line 1", "cpu-16 line 2"])
        await pilot.press("space", "space")
        assert shown(app)[1][1] == ["cpu-16 line 1", "cpu-16 line 2"]
    assert calls == ["cpu-16"]  # asked once, at the first opening


async def test_slash_narrows_the_list_to_the_names_holding_the_text() -> None:
    app = OpeningList("Hardware tiers", items([]))
    async with app.run_test() as pilot:
        await pilot.press("/", "c", "p", "u")
        assert [label for label, _ in shown(app)] == ["cpu-8", "cpu-16"]
        await pilot.press("escape")
        assert [label for label, _ in shown(app)] == ["cpu-8", "cpu-16", "gpu-1"]


async def test_slash_matches_the_name_of_an_item_and_not_the_rest_of_its_label() -> (
    None
):
    labelled = [
        OpeningItem("cpu-8    id small-k8s", lambda: [], name="cpu-8"),
        OpeningItem("gpu-1    id gpu-k8s", lambda: [], name="gpu-1"),
    ]
    app = OpeningList("Hardware tiers", labelled)
    async with app.run_test() as pilot:
        await pilot.press("/", "k", "8", "s")
        assert shown(app) == []
        await pilot.press("escape", "/", "G", "P", "U")
        assert [label for label, _ in shown(app)] == ["gpu-1    id gpu-k8s"]


async def test_q_quits() -> None:
    app = OpeningList("Hardware tiers", items([]))
    async with app.run_test() as pilot:
        await pilot.press("q")
    assert app.return_code == 0


async def test_an_item_that_starts_open_shows_its_lines_at_once() -> None:
    calls: list[str] = []

    def lines() -> Sequence[str]:
        calls.append("cpu-8")
        return ["cores: 8", "memory: 32 GiB"]

    app = OpeningList("Hardware tiers", [OpeningItem("cpu-8", lines, starts_open=True)])
    async with app.run_test():
        assert shown(app) == [("cpu-8", ["cores: 8", "memory: 32 GiB"])]
    assert calls == ["cpu-8"]


async def test_an_item_whose_lines_fail_shows_the_failure_and_the_list_stays_up() -> (
    None
):
    def failing() -> Sequence[str]:
        raise DominoDeployError("the route answered HTTP 404")

    listed = [OpeningItem("qp-rag", failing), *items([])]
    app = OpeningList("Environments", listed)
    async with app.run_test() as pilot:
        await pilot.press("space")
        assert shown(app)[0] == ("qp-rag", ["the route answered HTTP 404"])
        await pilot.press("down")
        assert app.query_one(Tree).cursor_line == 1


async def test_the_line_of_keys_is_shown() -> None:
    app = OpeningList("Hardware tiers", items([]))
    async with app.run_test():
        text = " ".join(str(static.render()) for static in app.query(Static))
    for keys in ("up/down move", "space open/close", "/ filter", "q quit"):
        assert keys in text
