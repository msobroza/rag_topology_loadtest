"""The LangGraph workflow (``embed -> retrieve -> rerank -> generate``).

``RagWorkflow`` sees only ports, so the same graph runs in every topology; which
adapters sit behind them is the composition root's business. Retrieval mode, sizes and
prompts come in as one ``WorkflowParameters``. Each node records its wall time
in ``timings_ms`` (a LangGraph reducer merges the per-stage keys).

The retrieve stage finds candidates by vector (``dense``), by words
(``lexical``), or both at once, fused by reciprocal rank (``hybrid``). A lexical
workflow needs no vector, so its graph starts at retrieve and ``embed`` is 0 ms.

Example:
    >>> async with CompositionRoot(settings) as composition:
    ...     deps = await composition.workflow_dependencies()
    ...     workflow = RagWorkflow(deps, settings.workflow_parameters())
    ...     result = await workflow.run("How many vacation days?", mode="retrieve")
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from ..adapters.vector_store.base_vector_store import BaseLexicalSearch
from ..domain.passages import STAGES, RagMode, RagResult, ScoredPassage
from ..domain.rank_fusion import reciprocal_rank_fusion
from ..domain.workflow_parameters import WorkflowParameters
from .rag_dependencies import RagDependencies


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


# The retrieve stage's search for one retrieval mode: state in, candidates out.
CandidateFinder = Callable[[RagState], Awaitable[list[ScoredPassage]]]


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


def required_lexical_search(
    search: BaseLexicalSearch | None, mode: str
) -> BaseLexicalSearch:
    """Return the lexical search a lexical or hybrid workflow needs, or refuse.

    Args:
        search: The store's lexical search; None when the store ranks no text.
        mode: The retrieval mode that needs it, named in the error.

    Returns:
        ``search``, known not to be None.

    Raises:
        ValueError: ``search`` is None.

    Example:
        >>> required_lexical_search(deps.lexical_search, "hybrid")
    """
    if search is not None:
        return search
    raise ValueError(
        f"retrieval_mode={mode!r} needs a store that ranks text, but the vector "
        "store ranks no text: set RAG_VECTOR_STORE_BACKEND=elasticsearch, or "
        "RAG_RETRIEVAL_MODE=dense"
    )


class RagWorkflow:
    """A compiled graph over ``RagDependencies``; build once, ``run`` per request.

    Args:
        deps: The ports the nodes call.
        parameters: Retrieval mode, sizes and prompts.

    Raises:
        ValueError: The retrieval mode is ``lexical`` or ``hybrid`` and
            ``deps.lexical_search`` is None.
    """

    def __init__(self, deps: RagDependencies, parameters: WorkflowParameters) -> None:
        self.deps, self.parameters = deps, parameters
        self._find_candidates = self._candidate_finder()
        # ty 0.0.100 does not match a TypedDict class against LangGraph's
        # TypedDictLike protocol (its __required_keys__); mypy accepts it.
        builder = StateGraph(RagState)  # ty: ignore[invalid-argument-type]
        builder.add_node("retrieve", timed("retrieve", self._retrieve))
        builder.add_node("rerank", timed("rerank", self._rerank))
        builder.add_node("generate", timed("generate", self._generate))
        if parameters.retrieval_mode == "lexical":  # no vector to compute
            builder.add_edge(START, "retrieve")
        else:
            builder.add_node("embed", timed("embed", self._embed))
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
        query = self.parameters.query_prefix + state["question"]
        [vector] = await self.deps.embedder.embed([query])
        return {"query_embedding": vector}

    async def _retrieve(self, state: RagState) -> dict[str, Any]:
        return {"candidates": await self._find_candidates(state)}

    # Chosen once at construction, so a lexical or hybrid workflow over a store
    # that ranks no text fails at start-up, not on its first request.
    def _candidate_finder(self) -> CandidateFinder:
        store, top_k = self.deps.vector_store, self.parameters.top_k_retrieve
        mode, rrf_k = self.parameters.retrieval_mode, self.parameters.rrf_k

        async def dense(state: RagState) -> list[ScoredPassage]:
            return await store.query(state["query_embedding"], top_k)

        if mode == "dense":
            return dense
        search = required_lexical_search(self.deps.lexical_search, mode)

        async def lexical(state: RagState) -> list[ScoredPassage]:
            return await search.lexical_query(state["question"], top_k)

        if mode == "lexical":
            return lexical

        async def hybrid(state: RagState) -> list[ScoredPassage]:
            # Both legs at once, each asked for top_k, fused back to top_k.
            rankings = await asyncio.gather(dense(state), lexical(state))
            return reciprocal_rank_fusion(rankings, k=rrf_k, top_k=top_k)

        return hybrid

    async def _rerank(self, state: RagState) -> dict[str, Any]:
        reranked = await self.deps.reranker.rerank(
            state["question"], state["candidates"], self.parameters.top_k_rerank
        )
        return {"reranked": reranked}

    async def _generate(self, state: RagState) -> dict[str, Any]:
        prompts = self.parameters.prompts
        messages = prompts.messages(state["question"], state["reranked"])
        return {"answer": await self.deps.llm.generate(messages)}
