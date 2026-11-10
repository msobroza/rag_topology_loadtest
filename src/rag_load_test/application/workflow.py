"""The LangGraph workflow (``embed -> retrieve -> rerank -> generate``).

``RagWorkflow`` sees only ports, so the same graph runs in every topology; which
adapters sit behind them is ``RagPorts``' business. Retrieval sizes and prompts
come in as one ``WorkflowParameters``. Each node records its wall time in
``timings_ms`` (a LangGraph reducer merges the per-stage keys).

Example:
    >>> async with RagPorts(settings) as ports:
    ...     deps = await ports.workflow_dependencies()
    ...     workflow = RagWorkflow(deps, settings.workflow_parameters())
    ...     result = await workflow.run("How many vacation days?", mode="retrieve")
"""

from __future__ import annotations

import time
from collections.abc import Awaitable
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from ..domain.passages import STAGES, RagMode, RagResult, ScoredPassage
from ..domain.ports import RagDependencies
from ..domain.workflow_parameters import WorkflowParameters


def merge_timings(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """Merge two stage-timing dicts; the LangGraph reducer of ``timings_ms``.

    Args:
        left: Timings recorded so far.
        right: Timings from the latest update; they win on a shared key.

    Returns:
        A new dict with the keys of both.
    """
    return {**left, **right}


class RagState(TypedDict, total=False):
    """The graph state one run carries from node to node."""

    question: str
    mode: str
    query_embedding: list[float]
    candidates: list[ScoredPassage]
    reranked: list[ScoredPassage]
    answer: str | None
    timings_ms: Annotated[dict[str, float], merge_timings]


class Node(Protocol):
    """An async graph node: ``state`` in, partial state update out.

    A Protocol rather than ``Callable[[RagState], ...]`` because LangGraph 1.x
    types nodes by a parameter named ``state``; a ``Callable`` alias erases the
    name and mypy then rejects ``StateGraph.add_node``.
    """

    def __call__(self, state: RagState) -> Awaitable[dict[str, Any]]:
        """Run the node.

        Args:
            state: The graph state so far.

        Returns:
            An awaitable of the partial state update.
        """


def timed(stage: str, node: Node) -> Node:
    """Wrap a node so its update also carries ``{"timings_ms": {stage: ms}}``.

    Args:
        stage: The key the node's wall time is recorded under.
        node: The node to time.

    Returns:
        The timed node.
    """

    async def run(state: RagState) -> dict[str, Any]:
        started = time.perf_counter()
        update = await node(state)
        return {**update, "timings_ms": {stage: (time.perf_counter() - started) * 1000}}

    return run


class RagWorkflow:
    """A compiled graph over ``RagDependencies``; build once, ``run`` per request.

    Args:
        deps: The ports the nodes call.
        parameters: Retrieval sizes and prompts.
    """

    def __init__(self, deps: RagDependencies, parameters: WorkflowParameters) -> None:
        self.deps, self.parameters = deps, parameters
        builder = StateGraph(RagState)
        builder.add_node("embed", timed("embed", self._embed))
        builder.add_node("retrieve", timed("retrieve", self._retrieve))
        builder.add_node("rerank", timed("rerank", self._rerank))
        builder.add_node("generate", timed("generate", self._generate))
        builder.add_edge(START, "embed")
        builder.add_edge("embed", "retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_conditional_edges(
            "rerank",
            lambda state: "generate" if state.get("mode") == "query" else END,
            {"generate": "generate", END: END},
        )
        builder.add_edge("generate", END)
        self.graph = builder.compile()

    async def run(self, question: str, *, mode: RagMode = "query") -> RagResult:
        """Invoke the graph once.

        Args:
            question: The question to answer.
            mode: ``query`` runs every stage; ``retrieve`` stops after reranking
                and leaves ``answer`` None.

        Returns:
            The result; ``timings_ms`` has one key per stage (0.0 for a skipped
            one) and ``total``, the wall time around the whole graph.
        """
        state: RagState = {"question": question, "mode": mode, "timings_ms": {}}
        started = time.perf_counter()
        final = await self.graph.ainvoke(state)
        timings = {stage: final["timings_ms"].get(stage, 0.0) for stage in STAGES}
        timings["total"] = (time.perf_counter() - started) * 1000
        return RagResult(
            question=question,
            mode=mode,
            answer=final.get("answer") if mode == "query" else None,
            passages=final.get("reranked", []),
            timings_ms=timings,
        )

    async def _embed(self, state: RagState) -> dict[str, Any]:
        [vector] = await self.deps.embedder.embed([state["question"]])
        return {"query_embedding": vector}

    async def _retrieve(self, state: RagState) -> dict[str, Any]:
        hits = await self.deps.vector_store.query(
            state["query_embedding"], self.parameters.top_k_retrieve
        )
        return {"candidates": hits}

    async def _rerank(self, state: RagState) -> dict[str, Any]:
        reranked = await self.deps.reranker.rerank(
            state["question"], state["candidates"], self.parameters.top_k_rerank
        )
        return {"reranked": reranked}

    async def _generate(self, state: RagState) -> dict[str, Any]:
        prompts = self.parameters.prompts
        messages = prompts.messages(state["question"], state["reranked"])
        return {"answer": await self.deps.chat_model.generate(messages)}
