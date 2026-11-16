"""Named fakes of the ports a stage calls, which record what the workflow asked them,
and a tracer that records the name of every step of a run.

Shared by the tests of the stage types, of the graphs and of the workflow app.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler

from query_processing.adapters.embedder.base_embedder import BaseEmbedder
from query_processing.adapters.reranker.base_reranker import BaseReranker
from query_processing.adapters.vector_store.base_vector_store import (
    BaseLexicalSearch,
    BaseVectorStore,
)
from query_processing.domain.metadata_filter import MetadataCondition
from query_processing.domain.passages import Passage, ScoredPassage


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
