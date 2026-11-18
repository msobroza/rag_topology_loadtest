"""The body of a ``/query`` or ``/retrieve`` request: question, history, filter.

One model for every reader of a body: the workflow app takes it from a client,
and the load test checks each line of its questions file against it, so a line
the app would answer with a 422 stops the load test when the file is read.

Example:
    >>> QueryRequest.model_validate({"question": "Vacation days?"}).metadata_filter
    []
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .conversation import HistoryMessage
from .retrieval.metadata_filter import MetadataCondition


class QueryRequest(BaseModel):
    """The question to answer, 1 to 2000 characters, its history and its conditions.

    ``conversational_history`` is optional: the earlier turns, in order, for the
    prompts to read. ``metadata_filter`` is optional: every passage retrieved
    meets all its conditions. Each condition, a history message's included, is
    checked against the fields the vector stores declare, by the app, which
    knows them.

    Example:
        >>> QueryRequest(question="And for a rented car?").conversational_history
        []
    """

    question: str = Field(min_length=1, max_length=2000)
    conversational_history: list[HistoryMessage] = Field(default_factory=list)
    metadata_filter: list[MetadataCondition] = Field(default_factory=list)
