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

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from ..domain.conversation import HistoryMessage
from ..domain.metadata_filter import MetadataCondition
from ..domain.passages import RagMode, ScoredPassage

# What one request carries from stage to stage, by field name: a stage reads
# the values its reads name and returns the fields it writes, the update
# LangGraph merges into the state.
WorkflowState = dict[str, Any]

# The field that holds each stage's wall time under the stage's name; a
# LangGraph reducer, merge_by_stage, merges the stages' updates.
TIMINGS_FIELD = "timings_ms"
# The field that holds the tokens each stage's LLM read and wrote, under the
# stage's name, as {"prompt": n, "completion": m}; merged as the timings are.
# A stage hands its counts unnamed in its update, and timed() puts them under
# its name: a stage type never learns its own name.
TOKENS_FIELD = "tokens"
# The tokens one stage reports: "prompt" and "completion".
StageTokens = dict[str, int]
# The fields the workflow measures for each stage: no stage writes them.
MEASURED_FIELDS: tuple[str, ...] = (TIMINGS_FIELD, TOKENS_FIELD)
# The conditions a search must meet: the client's, then those stages add. The
# one field that accumulates; state_fields.add_conditions is its reducer.
METADATA_FILTER_FIELD = "metadata_filter"
# The earlier turns the request carries: a prompt's placeholder splices them.
CONVERSATIONAL_HISTORY_FIELD = "conversational_history"

# The type of a field of messages, which a prompt's placeholder splices.
MESSAGES_TYPE: Any = list[HistoryMessage]

# The fields the request writes before the first stage runs, with their types.
REQUEST_FIELD_TYPES: dict[str, Any] = {
    "conversational_query": str,  # the user's message as sent in this turn
    "standalone_query": str,  # the same request made self-contained; searched
    # The earlier turns, as the client sent them; the workflow keeps none.
    CONVERSATIONAL_HISTORY_FIELD: MESSAGES_TYPE,
    "mode": RagMode,
    METADATA_FILTER_FIELD: list[MetadataCondition],
}

# The fields the response reads, with the types of the result's attributes they
# fill: names fixed in the toolkit, as the wire's are.
RESPONSE_FIELD_TYPES: dict[str, Any] = {
    "answer": str | None,  # none when the run ends before an answer
    "reranked": list[ScoredPassage],  # the result's passages
    METADATA_FILTER_FIELD: list[MetadataCondition],
}
RESPONSE_FIELDS: frozenset[str] = frozenset(RESPONSE_FIELD_TYPES)

# What one stage measures: its wall time, or its tokens.
_Measure = TypeVar("_Measure")


def merge_by_stage(
    left: Mapping[str, _Measure], right: Mapping[str, _Measure]
) -> dict[str, _Measure]:
    """Merge two measures by stage; the reducer of ``timings_ms`` and ``tokens``.

    Args:
        left: The measures recorded so far, by stage name.
        right: The measures of the latest update; they win on a shared key.

    Returns:
        A new dict with the keys of both.

    Example:
        >>> merge_by_stage({"embed": 1.0}, {"retrieve": 2.0})
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
        The request's fields, and no timings nor tokens.

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
        **{field: {} for field in MEASURED_FIELDS},
    }


# The fields every request provides before the first stage runs: a stage may
# read them wherever it is.
REQUEST_FIELDS: frozenset[str] = frozenset({*REQUEST_FIELD_TYPES, *MEASURED_FIELDS})
