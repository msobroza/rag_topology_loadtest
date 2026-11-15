"""The workflow state: what a request starts with, and how stage timings merge."""

from __future__ import annotations

from rag_load_test.application.workflow_state import initial_state, merge_timings


def test_the_question_is_both_queries_and_the_history_is_empty() -> None:
    assert initial_state("vacation days?", "retrieve") == {
        "conversational_query": "vacation days?",
        "standalone_query": "vacation days?",
        "history": [],
        "mode": "retrieve",
        "timings_ms": {},
    }


def test_merge_timings_merges_dicts_right_wins() -> None:
    assert merge_timings({"embed": 1.0}, {"retrieve": 2.0}) == {
        "embed": 1.0,
        "retrieve": 2.0,
    }
    assert merge_timings({"embed": 1.0}, {"embed": 3.0}) == {"embed": 3.0}
    assert merge_timings({}, {}) == {}
