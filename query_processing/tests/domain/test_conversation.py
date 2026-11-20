"""A conversational history's messages, and their conditions checked."""

from __future__ import annotations

import pytest
from query_processing_core.retrieval.metadata_filter import (
    MetadataCondition,
    MetadataFieldInfo,
)

from query_processing.domain.conversation import (
    HistoryMessage,
    check_history_conditions,
)

TOPIC = MetadataFieldInfo(name="topic", type="string", description="d")


def _condition(attribute: str) -> MetadataCondition:
    return MetadataCondition(attribute=attribute, comparator="eq", value="pay")


def test_conditions_that_fit_pass() -> None:
    history = [
        HistoryMessage(role="user", content="a", metadata_filter=[_condition("topic")]),
        HistoryMessage(role="assistant", content="b"),
    ]

    check_history_conditions(history, [TOPIC])


def test_a_condition_that_does_not_fit_names_its_message_and_place() -> None:
    history = [
        HistoryMessage(role="user", content="a"),
        HistoryMessage(
            role="user",
            content="b",
            metadata_filter=[_condition("topic"), _condition("sorce")],
        ),
    ]

    with pytest.raises(
        ValueError, match=r"conversational_history\[1\]\.metadata_filter\[1\].*'sorce'"
    ):
        check_history_conditions(history, [TOPIC])
