"""The workflow state: what the request provides, and what the response reads.

The state is no closed class: ``state_fields.state_type`` builds it from the
fields the request and the stages write, each with the type of what writes it.
A request provides the conversational query, the conversational history (the
earlier turns the client sends, possibly none), the standalone query, the mode
and the metadata filter (the client's conditions, possibly none); each stage
reads some fields and writes others, as the workflow file states. The
standalone query is the question as asked, until a stage rewrites it.

Example:
    >>> state = initial_state("How many vacation days?", "retrieve")
    >>> state["standalone_query"]
    'How many vacation days?'
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..domain.conversation import HistoryMessage
from ..domain.metadata_filter import MetadataCondition
from ..domain.passages import RagMode

# What one request carries from stage to stage, by field name: a stage reads
# the values its reads name and returns the fields it writes, the update
# LangGraph merges into the state.
WorkflowState = dict[str, Any]

# The field that holds each stage's wall time under the stage's name; a
# LangGraph reducer, merge_timings, merges the stages' updates.
TIMINGS_FIELD = "timings_ms"
# The conditions a search must meet: the client's, then those stages add. The
# one field that accumulates; state_fields.add_conditions is its reducer.
METADATA_FILTER_FIELD = "metadata_filter"
# The earlier turns the request carries: a prompt's placeholder splices them.
CONVERSATIONAL_HISTORY_FIELD = "conversational_history"

# The fields the request writes before the first stage runs, with their types.
REQUEST_FIELD_TYPES: dict[str, Any] = {
    "conversational_query": str,  # the user's message as sent in this turn
    "standalone_query": str,  # the same request made self-contained; searched
    # The earlier turns, as the client sent them; the workflow keeps none.
    CONVERSATIONAL_HISTORY_FIELD: list[HistoryMessage],
    "mode": RagMode,
    METADATA_FILTER_FIELD: list[MetadataCondition],
}

# The fields the response reads: names fixed in the toolkit, as the wire's are.
RESPONSE_FIELDS: frozenset[str] = frozenset(
    {"answer", "reranked", METADATA_FILTER_FIELD}
)


def merge_timings(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Merge two stage-timing dicts; the LangGraph reducer of ``timings_ms``.

    Args:
        left: Timings recorded so far.
        right: Timings from the latest update; they win on a shared key.

    Returns:
        A new dict with the keys of both.

    Example:
        >>> merge_timings({"embed": 1.0}, {"retrieve": 2.0})
        {'embed': 1.0, 'retrieve': 2.0}
    """
    return {**left, **right}


def initial_state(
    question: str,
    mode: RagMode,
    *,
    conversational_history: Sequence[HistoryMessage] = (),
    metadata_filter: Sequence[MetadataCondition] = (),
) -> WorkflowState:
    """Return the state one request starts with.

    Args:
        question: The request's question, copied into the conversational query
            and, until a stage rewrites it, the standalone query.
        mode: How far the request runs through the stages.
        conversational_history: The earlier turns the client sent, in order.
        metadata_filter: The client's conditions, which every search meets.

    Returns:
        The request's fields and no timings.

    Example:
        >>> initial_state("vacation days?", "query")["conversational_history"]
        []
    """
    return {
        "conversational_query": question,
        "standalone_query": question,
        CONVERSATIONAL_HISTORY_FIELD: list(conversational_history),
        "mode": mode,
        METADATA_FILTER_FIELD: list(metadata_filter),
        TIMINGS_FIELD: {},
    }


# The fields every request provides before the first stage runs: a stage may
# read them wherever it is.
REQUEST_FIELDS: frozenset[str] = frozenset({*REQUEST_FIELD_TYPES, TIMINGS_FIELD})
