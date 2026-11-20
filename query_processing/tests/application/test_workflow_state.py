"""The workflow state: what a request starts with, and how stage measures merge."""

from __future__ import annotations

from query_processing_core.retrieval.metadata_filter import MetadataCondition

from query_processing.application.workflow_state import (
    initial_state,
    sum_timings_by_stage,
    sum_tokens_by_stage,
)
from query_processing.domain.conversation import HistoryMessage

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


def test_the_timings_of_two_stages_are_kept_and_those_of_one_stage_added() -> None:
    assert sum_timings_by_stage({"embed": 1.0}, {"retrieve": 2.0}) == {
        "embed": 1.0,
        "retrieve": 2.0,
    }
    assert sum_timings_by_stage({"embed": 1.0}, {"embed": 3.0}) == {"embed": 4.0}
    assert sum_timings_by_stage({}, {}) == {}


def test_the_tokens_of_one_stage_are_added_count_by_count() -> None:
    early = {"generate": {"prompt": 100, "completion": 7}}
    late = {"generate": {"prompt": 100, "completion": 20}}

    assert sum_tokens_by_stage(early, late) == {
        "generate": {"prompt": 200, "completion": 27}
    }
    assert sum_tokens_by_stage(early, {}) == early


def test_a_reducer_leaves_what_it_is_handed_as_it_was() -> None:
    recorded = {"generate": {"prompt": 1, "completion": 1}}

    sum_tokens_by_stage(recorded, recorded)

    assert recorded == {"generate": {"prompt": 1, "completion": 1}}
