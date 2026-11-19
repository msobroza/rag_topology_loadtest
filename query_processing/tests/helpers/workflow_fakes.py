"""Named fakes of the ports a stage calls, which record what the workflow asked them,
a tracer that records the name of every step of a run, which it reaches
through LangGraph's compiled graph (``compiled_langgraph``), and an orchestrator
that refuses a graph, registered for one test (``register_orchestrator``).

Shared by the tests of the stage types, of the graphs and of the workflow app.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from query_processing.adapters.embedder.base_embedder import BaseEmbedder
from query_processing.adapters.llm.base_llm import BaseLLM
from query_processing.adapters.orchestrator.base_orchestrator import (
    ORCHESTRATORS,
    BaseOrchestrator,
)
from query_processing.adapters.orchestrator.langgraph.langgraph_orchestrator import (
    LangGraphOrchestrator,
)
from query_processing.adapters.reranker.base_reranker import BaseReranker
from query_processing.adapters.vector_store.base_vector_store import (
    BaseLexicalSearch,
    BaseVectorStore,
)
from query_processing.application.workflow import CompiledWorkflow
from query_processing.domain.components.component_requests import GenerateAnswer
from query_processing.domain.graph_description import GraphDescription
from query_processing.domain.opened_graph import OpenedGraph
from query_processing.domain.retrieval.metadata_filter import MetadataCondition
from query_processing.domain.retrieval.passages import Passage, ScoredPassage


def compiled_langgraph(
    workflow: CompiledWorkflow,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """LangGraph's compiled graph, of a workflow built on the langgraph orchestrator."""
    orchestrator = workflow.orchestrator
    assert isinstance(orchestrator, LangGraphOrchestrator), orchestrator
    return orchestrator.graph


class RefusingOrchestrator(BaseOrchestrator):
    """Refuses a graph with a conditional edge, naming its source; is never built."""

    @classmethod
    def check_graph_runs(cls, graph: GraphDescription) -> None:
        for edge in graph.conditional_edges:
            raise ValueError(
                f"stage {edge.source!r} has a conditional edge, which this library "
                "does not run (expected plain edges only)"
            )

    def __init__(self, opened: OpenedGraph) -> None:
        raise AssertionError("built, though start-up asks its check first")

    async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("ran, though it is never built")


def register_orchestrator(
    monkeypatch: pytest.MonkeyPatch, name: str, orchestrator: type[BaseOrchestrator]
) -> None:
    """Register ``orchestrator`` under ``name`` in ``ORCHESTRATORS`` until the test ends.

    The registry takes no name back, so its entries are put back as they were:
    a name left registered would change what an unknown one's refusal lists.
    """
    monkeypatch.setattr(ORCHESTRATORS, "_entries", dict(ORCHESTRATORS._entries))
    ORCHESTRATORS.register(name)(orchestrator)


def scored(passage_id: str, text: str | None = None) -> ScoredPassage:
    """A retrieved passage whose text is its id unless one is given."""
    return ScoredPassage(id=passage_id, text=text or passage_id, retrieval_score=1.0)


class RecordingEmbedder(BaseEmbedder):
    """Answers a fixed vector and records every query and passage it embedded."""

    model_name = "recording-embedder"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.passages: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.passages.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]

    async def ready(self) -> tuple[bool, str]:
        return True, self.model_name


class RecordingReranker(BaseReranker):
    """Keeps retrieval order and records ``(query, candidates, top_n)`` per call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        self.calls.append((query, len(passages), top_n))
        return list(passages[:top_n])

    async def ready(self) -> tuple[bool, str]:
        return True, "recording"


class CapRecordingLLM(BaseLLM):
    """Answers "ok" and records the answer cap each ``generate`` is given."""

    model_name = "cap-recording-llm"

    def __init__(self) -> None:
        self.caps: list[int | None] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None = None,
        max_tokens: int | None = None,
    ) -> GenerateAnswer:
        self.caps.append(max_tokens)
        return GenerateAnswer(text="ok")


class DenseOnlyStore(BaseVectorStore):
    """A store that ranks no text, as open-source Chroma: dense queries only."""

    def __init__(self, dense: Sequence[str] = ()) -> None:
        self.dense = list(dense)

    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        raise AssertionError("the workflow never writes to the store")

    async def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> list[ScoredPassage]:
        return [scored(pid) for pid in self.dense[:top_k]]

    async def count(self) -> int:
        return len(self.dense)

    async def ready(self) -> tuple[bool, str]:
        return True, "dense-only"


class ScriptedStore(BaseVectorStore, BaseLexicalSearch):
    """Answers one fixed ranking per kind of query and records the calls."""

    def __init__(self, dense: Sequence[str], lexical: Sequence[str]) -> None:
        self.dense, self.lexical = list(dense), list(lexical)
        self.calls: list[tuple[str, object, int]] = []
        # The metadata filter of each call, in the order of the calls.
        self.filters: list[list[MetadataCondition]] = []

    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        raise AssertionError("the workflow never writes to the store")

    async def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> list[ScoredPassage]:
        self.calls.append(("dense", list(embedding), top_k))
        self.filters.append(list(metadata_filter))
        return [scored(pid) for pid in self.dense[:top_k]]

    async def lexical_query(
        self,
        text: str,
        top_k: int,
        metadata_filter: Sequence[MetadataCondition] = (),
    ) -> list[ScoredPassage]:
        self.calls.append(("lexical", text, top_k))
        self.filters.append(list(metadata_filter))
        return [scored(pid) for pid in self.lexical[:top_k]]

    async def count(self) -> int:
        return len({*self.dense, *self.lexical})

    async def ready(self) -> tuple[bool, str]:
        return True, "scripted"


class ChainNameRecorder(AsyncCallbackHandler):
    """Records the name each step of a run reports to a tracer when it starts."""

    def __init__(self) -> None:
        self.names: list[str | None] = []

    async def on_chain_start(
        self, serialized: dict[str, Any] | None, inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        del serialized, inputs  # LangChain passes both; only the name is recorded
        self.names.append(kwargs.get("name"))
