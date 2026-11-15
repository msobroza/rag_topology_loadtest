"""``CompiledWorkflow``: one request through a compiled graph, its result out.

``workflow_builder.build_workflow`` compiles the graph from its description;
this module only runs it. The request's question starts the workflow state,
and the final state becomes the result: the answer in ``query`` mode, the
reranked passages, and each stage's wall time, keyed by the stages of the
graph that ran.

Example:
    >>> workflow = build_workflow(settings.workflow_graph(), deps.components())
    >>> result = await workflow.run("How many vacation days?", mode="retrieve")
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from ..domain.passages import TOTAL_TIMING_KEY, RagMode, RagResult
from .workflow_state import initial_state


class CompiledWorkflow:
    """A compiled graph; build it once, ``run`` it per request.

    Args:
        graph: The compiled LangGraph graph over the workflow state.
        stage_names: The graph's stages, in the order of its description: the
            keys of every result's ``timings_ms``, before the total.
    """

    # Any, not WorkflowState: ty 0.0.100 does not match a TypedDict against
    # LangGraph's TypedDictLike bound (its __optional_keys__); mypy accepts it.
    def __init__(
        self, graph: CompiledStateGraph[Any, Any, Any, Any], stage_names: Sequence[str]
    ) -> None:
        self.graph = graph
        self.stage_names = tuple(stage_names)

    async def run(self, question: str, *, mode: RagMode = "query") -> RagResult:
        """Invoke the graph once.

        Args:
            question: The question to answer.
            mode: ``query`` runs every stage; ``retrieve`` stops after reranking
                and leaves ``answer`` None.

        Returns:
            The result; ``timings_ms`` is ``stage_timings`` of what the stages
            recorded and the wall time around the whole graph.

        Example:
            >>> (await workflow.run("vacation days?", mode="retrieve")).answer is None
            True
        """
        started = time.perf_counter()
        final = await self.graph.ainvoke(initial_state(question, mode))
        total_ms = (time.perf_counter() - started) * 1000
        return RagResult(
            question=question,
            mode=mode,
            answer=final.get("answer") if mode == "query" else None,
            passages=final.get("reranked", []),
            timings_ms=stage_timings(self.stage_names, final["timings_ms"], total_ms),
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
