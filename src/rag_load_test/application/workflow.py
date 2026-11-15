"""``CompiledWorkflow``: one request through a compiled graph, its result out.

``workflow_builder.build_workflow`` compiles the graph from its description;
this module only runs it. The request's question starts the workflow state,
and the final state becomes the result: the answer in ``query`` mode, the
reranked passages, and each stage's wall time.

Example:
    >>> workflow = build_workflow(settings.workflow_graph(), deps.components())
    >>> result = await workflow.run("How many vacation days?", mode="retrieve")
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from ..domain.passages import STAGES, RagMode, RagResult
from .workflow_state import initial_state


class CompiledWorkflow:
    """A compiled graph; build it once, ``run`` it per request.

    Args:
        graph: The compiled LangGraph graph over the workflow state.
    """

    # Any, not WorkflowState: ty 0.0.100 does not match a TypedDict against
    # LangGraph's TypedDictLike bound (its __optional_keys__); mypy accepts it.
    def __init__(self, graph: CompiledStateGraph[Any, Any, Any, Any]) -> None:
        self.graph = graph

    async def run(self, question: str, *, mode: RagMode = "query") -> RagResult:
        """Invoke the graph once.

        Args:
            question: The question to answer.
            mode: ``query`` runs every stage; ``retrieve`` stops after reranking
                and leaves ``answer`` None.

        Returns:
            The result; ``timings_ms`` has one key per stage of ``STAGES`` (0.0
            for one that did not run) and ``total``, the wall time around the
            whole graph.

        Example:
            >>> (await workflow.run("vacation days?", mode="retrieve")).answer is None
            True
        """
        started = time.perf_counter()
        final = await self.graph.ainvoke(initial_state(question, mode))
        timings = {stage: final["timings_ms"].get(stage, 0.0) for stage in STAGES}
        timings["total"] = (time.perf_counter() - started) * 1000
        return RagResult(
            question=question,
            mode=mode,
            answer=final.get("answer") if mode == "query" else None,
            passages=final.get("reranked", []),
            timings_ms=timings,
        )
