"""How the workflow state takes a stage's update, stated once by the opened graph.

A field with a reducer is reduced, any other is replaced, and a field the
update does not hold is kept. The reducers here are plain functions: the
domain is handed them as values.
"""

from __future__ import annotations

from typing import Any

from query_processing.domain.opened_graph import OpenedGraph


def appended(left: list[str], right: list[str]) -> list[str]:
    return [*left, *right]


def opened_with(reducers: dict[str, Any]) -> OpenedGraph:
    """An opened graph of no stage, with these reducers."""
    return OpenedGraph(
        stages={}, edges=[], conditional_edges=[], field_types={}, reducers=reducers
    )


def test_a_field_with_a_reducer_is_reduced() -> None:
    opened = opened_with({"notes": appended})

    merged = opened.merged({"notes": ["first"]}, {"notes": ["second"]})

    assert merged == {"notes": ["first", "second"]}


def test_a_field_without_a_reducer_is_replaced() -> None:
    opened = opened_with({"notes": appended})

    merged = opened.merged({"draft": "old"}, {"draft": "new"})

    assert merged == {"draft": "new"}


def test_a_field_the_update_does_not_hold_is_kept() -> None:
    opened = opened_with({"notes": appended})

    merged = opened.merged({"draft": "old", "notes": ["first"]}, {"answer": "yes"})

    assert merged == {"draft": "old", "notes": ["first"], "answer": "yes"}


def test_the_values_handed_in_are_not_changed() -> None:
    values = {"draft": "old", "notes": ["first"]}

    opened_with({"notes": appended}).merged(values, {"draft": "new", "notes": ["x"]})

    assert values == {"draft": "old", "notes": ["first"]}
