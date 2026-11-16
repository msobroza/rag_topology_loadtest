"""The toolkit's edge conditions: functions of the values their edge reads."""

from __future__ import annotations

import collections.abc
from typing import TYPE_CHECKING, Any, Literal

import pytest

from query_processing.application.edge_conditions.edge_condition import (
    EDGE_CONDITIONS,
    answers_with_label,
    taken_types,
)
from query_processing.application.edge_conditions.has_history import has_history
from query_processing.application.edge_conditions.is_query_mode import is_query_mode
from query_processing.application.edge_conditions.label import label
from query_processing.domain.conversation import HistoryMessage

if TYPE_CHECKING:  # never imported at run time: an annotation of it cannot be read
    from collections.abc import Sequence


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


# --- what an edge condition answers, and what it takes -------------------------------


def unannotated(value):  # type: ignore[no-untyped-def]
    return value


def optional_label(value: str) -> str | None:
    return value


def literal_label(value: str) -> Literal["a", "b"]:
    return "a" if value == "a" else "b"


def typed_under_type_checking(history: Sequence[object]) -> bool:
    return bool(history)


def test_a_literal_of_strings_answers_with_a_label() -> None:
    assert answers_with_label(literal_label)


@pytest.mark.parametrize(
    ("condition", "said"),
    [
        (unannotated, "states no return type"),
        (optional_label, "is annotated to return str | None"),
    ],
)
def test_an_edge_condition_that_answers_neither_yes_or_no_nor_a_label_is_refused(
    condition: Any, said: str
) -> None:
    with pytest.raises(ValueError, match=said) as refused:
        answers_with_label(condition)

    assert "(expected -> bool for a yes-or-no answer, or -> str" in str(refused.value)


def test_annotations_that_cannot_be_read_are_refused_with_what_to_do() -> None:
    with pytest.raises(ValueError, match="cannot be read") as refused:
        answers_with_label(typed_under_type_checking)

    assert str(refused.value) == (
        "the annotations of the edge condition 'typed_under_type_checking' "
        "cannot be read: NameError: name 'Sequence' is not defined (expected "
        "annotations of names the module imports at run time, not under "
        "TYPE_CHECKING)"
    )


def test_the_types_an_edge_condition_takes_are_those_of_its_parameters() -> None:
    assert taken_types(has_history) == [collections.abc.Sequence[object]]
    assert taken_types(unannotated) == [Any]
