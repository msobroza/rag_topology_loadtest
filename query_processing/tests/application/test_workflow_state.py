"""The workflow state: what a request starts with, and how stage measures merge."""

from __future__ import annotations

from query_processing.application.workflow_state import initial_state, merge_by_stage
from query_processing.domain.conversation import HistoryMessage
from query_processing.domain.metadata_filter import MetadataCondition

ON_PAY = MetadataCondition(attribute="topic", comparator="eq", value="pay")


def test_the_question_is_both_queries_and_the_history_is_empty() -> None:
    assert initial_state("vacation days?", "retrieve") == {
        "conversational_query": "vacation days?",
        "standalone_query": "vacation days?",
        "conversational_history": [],
        "mode": "retrieve",
        "metadata_filter": [],
        "timings_ms": {},
        "tokens": {},
    }


def test_the_clients_conditions_start_the_metadata_filter() -> None:
    state = initial_state("vacation days?", "query", metadata_filter=(ON_PAY,))

    assert state["metadata_filter"] == [ON_PAY]


def test_the_clients_earlier_turns_are_the_conversational_history() -> None:
    history = (
        HistoryMessage(role="user", content="a"),
        HistoryMessage(role="user", content="b"),
    )

    state = initial_state("vacation days?", "query", conversational_history=history)

    assert state["conversational_history"] == list(history)
    assert state["standalone_query"] == "vacation days?"


def test_merge_by_stage_merges_dicts_right_wins() -> None:
    assert merge_by_stage({"embed": 1.0}, {"retrieve": 2.0}) == {
        "embed": 1.0,
        "retrieve": 2.0,
    }
    assert merge_by_stage({"embed": 1.0}, {"embed": 3.0}) == {"embed": 3.0}
    assert merge_by_stage({}, {}) == {}
