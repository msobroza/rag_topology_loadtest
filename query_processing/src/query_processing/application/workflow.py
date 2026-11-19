"""``CompiledWorkflow``: one request through a compiled graph, its result out.

``workflow_builder.build_workflow`` builds the orchestrator's graph from its
description; this module only runs it. The request's question starts the workflow state,
and the final state becomes the result: the answer in ``query`` mode, the
reranked passages, and each stage's wall time, keyed by the stages of the
graph that ran.

Example:
    >>> workflow = build_workflow(
    ...     settings.graph, deps.components(), context, orchestrator)
    >>> result = await workflow.run("How many vacation days?", mode="retrieve")
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from ..adapters.orchestrator.base_orchestrator import BaseOrchestrator
from ..domain.conversation import HistoryMessage
from ..domain.retrieval.metadata_filter import MetadataCondition
from ..domain.retrieval.passages import TOTAL_TIMING_KEY, RagMode, RagResult
from .workflow_state import (
    METADATA_FILTER_FIELD,
    TIMINGS_FIELD,
    TOKENS_FIELD,
    StageTokens,
    initial_state,
)


class CompiledWorkflow:
    """A compiled graph; build it once, ``run`` it per request.

    Args:
        orchestrator: The library's graph of the opened stages, which runs
            each request.
        stage_names: The graph's stages, in the order of its description: the
            keys of every result's ``timings_ms``, before the total.
    """

    def __init__(
        self, orchestrator: BaseOrchestrator, stage_names: Sequence[str]
    ) -> None:
        self.orchestrator = orchestrator
        self.stage_names = tuple(stage_names)

    async def run(
        self,
        question: str,
        *,
        mode: RagMode = "query",
        conversational_history: Sequence[HistoryMessage] = (),
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> RagResult:
        """Invoke the graph once.

        Args:
            question: The question to answer.
            mode: ``query`` runs every stage; ``retrieve`` stops after reranking
                and leaves ``answer`` None.
            conversational_history: The earlier turns the client sent.
            metadata_filter: The client's conditions, which every search meets.

        Returns:
            The result; ``timings_ms`` is ``stage_timings`` of what the stages
            recorded and the wall time around the whole graph, ``tokens`` the
            counts of each stage that reported them, in the graph's order, and
            ``metadata_filter`` the conditions applied, the client's first.

        Example:
            >>> (await workflow.run("vacation days?", mode="retrieve")).answer is None
            True
        """
        started = time.perf_counter()
        state = initial_state(
            question,
            mode,
            conversational_history=conversational_history,
            metadata_filter=metadata_filter,
        )
        final = await self.orchestrator.ainvoke(state)
        total_ms = (time.perf_counter() - started) * 1000
        return RagResult(
            question=question,
            mode=mode,
            answer=final.get("answer") if mode == "query" else None,
            passages=final.get("reranked", []),
            metadata_filter=final.get(METADATA_FILTER_FIELD, []),
            timings_ms=stage_timings(self.stage_names, final[TIMINGS_FIELD], total_ms),
            tokens=stage_tokens(self.stage_names, final[TOKENS_FIELD]),
        )


def stage_timings(
    stage_names: Sequence[str], recorded: Mapping[str, float], total_ms: float
) -> dict[str, float]:
    """Return a result's timings: every stage of the graph, then the total.

    A stage that did not run in this request keeps its key at 0.0, so every
    response of one graph has the same keys.

    Args:
        stage_names: The graph's stages, in the order of its description.
        recorded: Each stage that ran, by name, with its wall time in ms.
        total_ms: The wall time around the whole graph.

    Returns:
        One key per stage, in ``stage_names`` order, then ``TOTAL_TIMING_KEY``.

    Example:
        >>> stage_timings(["embed", "generate"], {"embed": 2.0}, 3.5)
        {'embed': 2.0, 'generate': 0.0, 'total': 3.5}
    """
    timings = {stage: recorded.get(stage, 0.0) for stage in stage_names}
    timings[TOTAL_TIMING_KEY] = total_ms
    return timings


def stage_tokens(
    stage_names: Sequence[str], recorded: Mapping[str, StageTokens]
) -> dict[str, StageTokens]:
    """Return a result's tokens: each stage that reported them, in the graph's order.

    Unlike the timings, a stage that reported none has no key: a missing count
    is never mistaken for 0.

    Args:
        stage_names: The graph's stages, in the order of its description.
        recorded: Each stage that reported tokens, by name, in the order it ran.

    Returns:
        The recorded tokens, in ``stage_names`` order.

    Example:
        >>> stage_tokens(["rewrite", "generate"],
        ...              {"generate": {"prompt": 9, "completion": 2}})
        {'generate': {'prompt': 9, 'completion': 2}}
    """
    return {stage: recorded[stage] for stage in stage_names if stage in recorded}
