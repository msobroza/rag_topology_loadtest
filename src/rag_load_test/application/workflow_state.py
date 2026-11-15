"""The workflow state: the one typed state every graph of the toolkit carries.

A request provides the conversational query, the standalone query, the history
and the mode; each stage reads some fields and writes others, as its stage type
declares. There is no history yet, so the standalone query is the question as
asked and the history is always empty: nothing reads it. A later stage type may
rewrite the conversational query into the standalone one, and the stages that
search keep reading the standalone query.

Example:
    >>> state = initial_state("How many vacation days?", "retrieve")
    >>> state["standalone_query"]
    'How many vacation days?'
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from ..domain.passages import RagMode, ScoredPassage


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


class WorkflowState(TypedDict, total=False):
    """What one request carries from stage to stage.

    ``timings_ms`` holds each stage's wall time under the stage's name; a
    LangGraph reducer merges the stages' updates.
    """

    conversational_query: str  # the user's message as sent in this turn
    standalone_query: str  # the same request made self-contained; what is searched
    history: list[tuple[str, str]]  # earlier turns as (message, answer); empty
    mode: RagMode
    query_embedding: list[float]
    candidates: list[ScoredPassage]
    reranked: list[ScoredPassage]
    answer: str | None
    timings_ms: Annotated[dict[str, float], merge_timings]


def initial_state(question: str, mode: RagMode) -> WorkflowState:
    """Return the state one request starts with.

    Args:
        question: The request's question, copied into the conversational query
            and, with no history to resolve it against, the standalone query.
        mode: How far the request runs through the stages.

    Returns:
        The request's fields, an empty history and no timings.

    Example:
        >>> initial_state("vacation days?", "query")["history"]
        []
    """
    return {
        "conversational_query": question,
        "standalone_query": question,
        "history": [],
        "mode": mode,
        "timings_ms": {},
    }
