"""The toolkit's edge conditions: functions of the values their edge reads."""

from __future__ import annotations

import pytest

from query_processing.application.edge_conditions.edge_condition import (
    EDGE_CONDITIONS,
    answers_with_label,
)
from query_processing.application.edge_conditions.has_history import has_history
from query_processing.application.edge_conditions.is_query_mode import is_query_mode
from query_processing.application.edge_conditions.label import label
from query_processing.domain.conversation import HistoryMessage


def test_the_three_edge_conditions() -> None:
    assert is_query_mode("query")
    assert not is_query_mode("retrieve")
    assert has_history([HistoryMessage(role="user", content="a")])
    assert not has_history([])
    assert label("leans_on_history") == "leans_on_history"


def test_the_three_are_registered_under_their_own_names() -> None:
    for condition in (is_query_mode, has_history, label):
        assert EDGE_CONDITIONS.get(condition.__name__) is condition


def test_only_label_answers_with_a_label() -> None:
    assert answers_with_label(label)
    assert not answers_with_label(is_query_mode)
    assert not answers_with_label(has_history)


@pytest.mark.parametrize(("mode", "expected"), [("query", True), ("retrieve", False)])
def test_is_query_mode_answers_from_the_mode_alone(mode: str, expected: bool) -> None:
    assert is_query_mode(mode) is expected
