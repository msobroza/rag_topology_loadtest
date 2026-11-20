"""The body of a ``/query`` or ``/retrieve`` request, one model for its readers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from query_processing_core.query_request import QueryRequest


def test_a_question_alone_is_a_body_with_no_history_and_no_filter() -> None:
    body = QueryRequest.model_validate({"question": "q?"})

    assert (body.conversational_history, body.metadata_filter) == ([], [])


@pytest.mark.parametrize("length", [1, 2000])
def test_a_question_of_1_to_2000_characters_is_accepted(length: int) -> None:
    assert len(QueryRequest(question="x" * length).question) == length


@pytest.mark.parametrize(
    ("body", "where"),
    [
        ({"question": ""}, "question"),
        ({"question": "x" * 2001}, "question"),
        ({"question": "q?", "conversational_history": [{"role": "system"}]}, "role"),
        ({"question": "q?", "metadata_filter": [{"attribute": "a"}]}, "comparator"),
    ],
    ids=["empty", "too-long", "a-bad-history", "a-bad-filter"],
)
def test_a_body_out_of_its_shape_is_refused(
    body: dict[str, object], where: str
) -> None:
    with pytest.raises(ValidationError, match=where):
        QueryRequest.model_validate(body)
