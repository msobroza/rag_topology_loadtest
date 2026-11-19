"""The workflow state: what the request provides, and what the response reads.

The state is no closed class: the orchestrator builds it from the
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

import operator
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from ..domain.conversation import HistoryMessage
from ..domain.retrieval.metadata_filter import MetadataCondition
from ..domain.retrieval.passages import RagMode, ScoredPassage

# What one request carries from stage to stage, by field name: a stage reads
# the values its reads name and returns the fields it writes, the update
# the orchestrator merges into the state.
WorkflowState = dict[str, Any]

# The field that holds each stage's wall time under the stage's name; its
# reducer, sum_timings_by_stage, adds the stages' updates, so a stage
# that runs more than once reports every run.
TIMINGS_FIELD = "timings_ms"
# The field that holds the tokens each stage's LLM read and wrote, under the
# stage's name, as {"prompt": n, "completion": m}; added as the timings are,
# by sum_tokens_by_stage.
# A stage hands its counts unnamed in its update, and timed() puts them under
# its name: a stage type never learns its own name.
TOKENS_FIELD = "tokens"
# The tokens one stage reports: "prompt" and "completion".
StageTokens = dict[str, int]
# The fields the workflow measures for each stage: no stage writes them.
MEASURED_FIELDS: tuple[str, ...] = (TIMINGS_FIELD, TOKENS_FIELD)
# The conditions a search must meet: the client's, then those stages add. The
# one field that accumulates; add_conditions is its reducer.
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


def sum_timings_by_stage(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, float]:
    """Add a stage's wall time to what it ran before; the reducer of ``timings_ms``.

    A stage runs more than once in a request in a loop, or where two branches
    join at different depths: its time is that of every run.

    Args:
        left: The times recorded so far, by stage name.
        right: The times of the latest update.

    Returns:
        A new dict with the keys of both, a shared key's times added.

    Example:
        >>> sum_timings_by_stage({"embed": 1.0, "draft": 2.0}, {"draft": 3.0})
        {'embed': 1.0, 'draft': 5.0}
    """
    return _sum_by_stage(left, right, operator.add)


def sum_tokens_by_stage(
    left: Mapping[str, StageTokens], right: Mapping[str, StageTokens]
) -> dict[str, StageTokens]:
    """Add a stage's tokens to what it ran before; the reducer of ``tokens``.

    Args:
        left: The tokens recorded so far, by stage name.
        right: The tokens of the latest update.

    Returns:
        A new dict with the keys of both, a shared key's counts added count by count.

    Example:
        >>> sum_tokens_by_stage({"draft": {"prompt": 9, "completion": 2}},
        ...                     {"draft": {"prompt": 9, "completion": 3}})
        {'draft': {'prompt': 18, 'completion': 5}}
    """
    return _sum_by_stage(left, right, _add_counts)


def _sum_by_stage(
    left: Mapping[str, _Measure],
    right: Mapping[str, _Measure],
    add: Callable[[_Measure, _Measure], _Measure],
) -> dict[str, _Measure]:
    # A new dict: LangGraph hands the reducer the channel's value, never a copy.
    summed = dict(left)
    for stage, measure in right.items():
        summed[stage] = add(summed[stage], measure) if stage in summed else measure
    return summed


def _add_counts(early: StageTokens, late: StageTokens) -> StageTokens:
    return {name: early.get(name, 0) + late.get(name, 0) for name in {**early, **late}}


def add_conditions(
    left: Sequence[MetadataCondition], right: Sequence[MetadataCondition]
) -> list[MetadataCondition]:
    """Add a stage's conditions to the metadata filter; the field's reducer.

    Also what makes the client's conditions the first state's, each once.

    Args:
        left: The conditions so far, the client's first.
        right: The conditions a stage writes.

    Returns:
        ``left``, then each condition of ``right`` it does not hold yet.

    Example:
        >>> add_conditions([on_pay], [on_pay, not_on_leave])
        [on_pay, not_on_leave]
    """
    added = list(left)
    for condition in right:
        if condition not in added:
            added.append(condition)
    return added


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
        The request's fields, and no timings nor tokens. The metadata filter
        holds each of the client's conditions once, as its reducer would make
        it: so no orchestrator has to reduce the first state for its result to
        be LangGraph's, which passes the first state through its reducers.

    Example:
        >>> initial_state("vacation days?", "query")["conversational_history"]
        []
    """
    return {
        "conversational_query": question,
        "standalone_query": question,
        CONVERSATIONAL_HISTORY_FIELD: list(conversational_history),
        "mode": mode,
        METADATA_FILTER_FIELD: add_conditions([], metadata_filter),
        **{field: {} for field in MEASURED_FIELDS},
    }


# The fields every request provides before the first stage runs: a stage may
# read them wherever it is.
REQUEST_FIELDS: frozenset[str] = frozenset({*REQUEST_FIELD_TYPES, *MEASURED_FIELDS})
