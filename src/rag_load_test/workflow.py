"""The LangGraph workflow (``embed -> retrieve -> rerank -> generate``) and its wiring.

``RagWorkflow`` sees only ports, so the same graph runs in every topology;
``build_dependencies`` picks the adapters from ``RagSettings``. Each node records
its wall time in ``timings_ms`` (a LangGraph reducer merges the per-stage keys).
Retrieval sizes are constants so experiments only vary the topology.

Example::

    deps = build_dependencies(settings, client=httpx_client, executor=pool)
    result = await RagWorkflow(deps).run("How many vacation days?", mode="retrieve")
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from .adapters import (
    CrossEncoderReranker,
    ElasticsearchVectorStore,
    NoReranker,
    OpenAIChatModel,
    SentenceTransformerEmbedder,
)
from .fakes import FakeChatModel, FakeEmbedder, FakeReranker
from .models import (
    STAGES,
    ChatModelPort,
    EmbedderPort,
    RagMode,
    RagResult,
    RerankerPort,
    ScoredPassage,
    VectorStorePort,
)
from .ovms import BearerToken, OvmsEmbedder, OvmsReranker
from .settings import RagSettings

TOP_K_RETRIEVE = 20  # candidates from the vector store
TOP_K_RERANK = 5  # passages kept after reranking and sent to the LLM
SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages. "
    "Cite passages as [n]. If the context is insufficient, say so."
)


def rag_messages(
    question: str, passages: Sequence[ScoredPassage]
) -> list[dict[str, str]]:
    """OpenAI-style messages: system prompt + numbered context + question."""
    context = "\n".join(f"[{i}] {p.text}" for i, p in enumerate(passages, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def merge_timings(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {**left, **right}


class RagState(TypedDict, total=False):
    question: str
    mode: str
    query_embedding: list[float]
    candidates: list[ScoredPassage]
    reranked: list[ScoredPassage]
    answer: str | None
    timings_ms: Annotated[dict[str, float], merge_timings]


@dataclass
class RagDependencies:
    embedder: EmbedderPort
    vector_store: VectorStorePort
    reranker: RerankerPort
    chat_model: ChatModelPort


Node = Callable[[RagState], Awaitable[dict[str, Any]]]


def timed(stage: str, node: Node) -> Node:
    """Wrap a node so its update also carries ``{"timings_ms": {stage: ms}}``."""

    async def run(state: RagState) -> dict[str, Any]:
        started = time.perf_counter()
        update = await node(state)
        return {**update, "timings_ms": {stage: (time.perf_counter() - started) * 1000}}

    return run


class RagWorkflow:
    """A compiled graph over ``RagDependencies``; build once, ``run`` per request."""

    def __init__(self, deps: RagDependencies) -> None:
        self.deps = deps
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
        """Invoke the graph once; ``timings_ms["total"]`` is wall time around it."""
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
            state["query_embedding"], TOP_K_RETRIEVE
        )
        return {"candidates": hits}

    async def _rerank(self, state: RagState) -> dict[str, Any]:
        reranked = await self.deps.reranker.rerank(
            state["question"], state["candidates"], TOP_K_RERANK
        )
        return {"reranked": reranked}

    async def _generate(self, state: RagState) -> dict[str, Any]:
        messages = rag_messages(state["question"], state["reranked"])
        return {"answer": await self.deps.chat_model.generate(messages)}


# --- wiring from settings ---------------------------------------------------


def build_embedder(
    settings: RagSettings, *, client: httpx.AsyncClient, executor: Executor
) -> EmbedderPort:
    if settings.embedder_backend == "fake":
        return FakeEmbedder()
    if settings.embedder_backend == "ovms":
        return OvmsEmbedder(
            client,
            settings.ovms_embeddings_url,
            settings.ovms_embeddings_model,
            token=BearerToken(client, settings.ovms_auth),
        )
    return SentenceTransformerEmbedder.from_pretrained(
        settings.embedder_model, executor
    )


def build_reranker(
    settings: RagSettings, *, client: httpx.AsyncClient, executor: Executor
) -> RerankerPort:
    if settings.reranker_backend == "none":
        return NoReranker()
    if settings.reranker_backend == "fake":
        return FakeReranker()
    if settings.reranker_backend == "ovms":
        return OvmsReranker(
            client,
            settings.ovms_rerank_url,
            settings.ovms_rerank_model,
            token=BearerToken(client, settings.ovms_auth),
        )
    return CrossEncoderReranker.from_pretrained(settings.reranker_model, executor)


def build_vector_store(settings: RagSettings) -> VectorStorePort:
    return ElasticsearchVectorStore.open(settings.es_url, api_key=settings.es_api_key)


def build_chat_model(settings: RagSettings) -> ChatModelPort:
    if settings.llm_backend == "fake":
        return FakeChatModel()
    return OpenAIChatModel.from_settings(settings)


def build_dependencies(
    settings: RagSettings, *, client: httpx.AsyncClient, executor: Executor
) -> RagDependencies:
    """Wire every port from settings; local models load their weights here."""
    return RagDependencies(
        embedder=build_embedder(settings, client=client, executor=executor),
        vector_store=build_vector_store(settings),
        reranker=build_reranker(settings, client=client, executor=executor),
        chat_model=build_chat_model(settings),
    )
