"""Adapters: sentence-transformers models, the Elasticsearch store, OpenAI.

sentence-transformers is synchronous, so model calls go through ``run_blocking``
on one bounded ``ThreadPoolExecutor``: the event loop stays free and overload
turns into queueing. Heavy imports live inside the methods that need them so
``import rag_load_test`` stays cheap for the OVMS and fake apps.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, TypeVar

from .models import Passage, RagDependencyError, ScoredPassage, top_by_score
from .settings import RagSettings

T = TypeVar("T")
MODEL_THREADS = 4  # thread pool for local model inference
INDEX = "rag_passages"  # Elasticsearch index holding the passages


def model_executor() -> ThreadPoolExecutor:
    """The shared pool for sentence-transformers inference."""
    return ThreadPoolExecutor(MODEL_THREADS, thread_name_prefix="rag-model")


async def run_blocking(executor: Executor, fn: Callable[..., T], *args: Any) -> T:
    """Await ``fn(*args)`` on ``executor`` (``functools.partial`` for kwargs)."""
    return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)


class SentenceTransformerEmbedder:
    """EmbedderPort on a ``SentenceTransformer``; inject a fake ``model`` in tests.

    Example::

        embedder = SentenceTransformerEmbedder.from_pretrained("BAAI/bge-small", pool)
        [vector] = await embedder.embed(["vacation policy"])
    """

    def __init__(
        self, model: Any, executor: Executor, *, model_name: str, batch_size: int = 64
    ) -> None:
        self._model, self._executor, self._batch_size = model, executor, batch_size
        self.model_name = model_name

    @classmethod
    def from_pretrained(
        cls, model_name: str, executor: Executor
    ) -> SentenceTransformerEmbedder:
        from sentence_transformers import SentenceTransformer  # pulls in torch

        model = SentenceTransformer(model_name, device="cpu")
        return cls(model, executor, model_name=model_name)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await run_blocking(self._executor, self._encode, list(texts))

    async def ready(self) -> tuple[bool, str]:
        return True, self.model_name

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Unit-norm vectors, so cosine similarity is a plain dot product.
        matrix = self._model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True
        )
        return matrix.tolist()


class CrossEncoderReranker:
    """RerankerPort on a sentence-transformers ``CrossEncoder`` (scores in 0..1).

    Example::

        reranker = CrossEncoderReranker.from_pretrained("BAAI/bge-reranker-base", pool)
        best = await reranker.rerank("vacation policy", candidates, top_n=5)
    """

    def __init__(
        self, model: Any, executor: Executor, *, model_name: str, batch_size: int = 32
    ) -> None:
        self._model, self._executor, self._batch_size = model, executor, batch_size
        self.model_name = model_name

    @classmethod
    def from_pretrained(
        cls, model_name: str, executor: Executor
    ) -> CrossEncoderReranker:
        from sentence_transformers import CrossEncoder  # pulls in torch

        model = CrossEncoder(model_name, device="cpu")
        return cls(model, executor, model_name=model_name)

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        if not passages:
            return []
        pairs = [(query, p.text) for p in passages]
        predict = functools.partial(
            self._model.predict, pairs, batch_size=self._batch_size
        )
        scores = await run_blocking(self._executor, predict)
        return top_by_score(passages, [float(s) for s in scores], top_n)

    async def ready(self) -> tuple[bool, str]:
        return True, self.model_name


class NoReranker:
    """RerankerPort that keeps retrieval order (``RAG_RERANKER_BACKEND=none``)."""

    async def rerank(
        self, query: str, passages: Sequence[ScoredPassage], top_n: int
    ) -> list[ScoredPassage]:
        return list(passages[:top_n])

    async def ready(self) -> tuple[bool, str]:
        return True, "none"


class ElasticsearchVectorStore:
    """VectorStorePort on LangChain's ``AsyncElasticsearchStore``.

    Vectors come from our own EmbedderPort, so the store is driven through
    ``aadd_embeddings`` and ``asimilarity_search_by_vector_with_relevance_scores``
    and never needs a LangChain ``Embeddings`` object. Any Elasticsearch failure
    surfaces as ``RagDependencyError("vector_store", ...)``.

    Example::

        store = ElasticsearchVectorStore.open("http://localhost:9200")
        hits = await store.query(vector, top_k=20)
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    @classmethod
    def open(cls, url: str, *, api_key: str = "") -> ElasticsearchVectorStore:
        from langchain_elasticsearch import AsyncElasticsearchStore

        return cls(
            AsyncElasticsearchStore(INDEX, es_url=url, es_api_key=api_key or None)
        )

    async def upsert(
        self, passages: Sequence[Passage], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if not passages:
            return
        pairs = [(p.text, list(e)) for p, e in zip(passages, embeddings, strict=True)]
        await self._call(
            self._store.aadd_embeddings,
            pairs,
            metadatas=[dict(p.metadata) for p in passages],
            ids=[p.id for p in passages],
        )

    async def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[ScoredPassage]:
        hits = await self._call(
            self._store.asimilarity_search_by_vector_with_relevance_scores,
            list(embedding),
            k=top_k,
            doc_builder=_document_with_id,
        )
        return [
            ScoredPassage(
                id=doc.id or "",
                text=doc.page_content,
                metadata=doc.metadata,
                retrieval_score=float(score),
            )
            for doc, score in hits
        ]

    async def count(self) -> int:
        client = self._store.client
        if not await self._call(client.indices.exists, index=INDEX):
            return 0
        return int((await self._call(client.count, index=INDEX))["count"])

    @staticmethod
    async def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            raise RagDependencyError("vector_store", detail, target=INDEX) from exc


def _document_with_id(hit: dict[str, Any]) -> Any:
    # The default builder drops Elasticsearch's _id, which is our passage id.
    from langchain_core.documents import Document

    source = hit["_source"]
    return Document(
        id=hit["_id"],
        page_content=source.get("text", ""),
        metadata=source.get("metadata", {}),
    )


class OpenAIChatModel:
    """ChatModelPort on the OpenAI Chat Completions API (``AsyncOpenAI``).

    The SDK reads ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` itself, so a Domino AI
    Gateway can be substituted without code changes.

    Example::

        answer = await OpenAIChatModel.from_settings(settings).generate(messages)
    """

    def __init__(self, client: Any, model: str) -> None:
        self._client, self.model_name = client, model

    @classmethod
    def from_settings(cls, settings: RagSettings) -> OpenAIChatModel:
        import openai  # SDK defaults: 10 min timeout, 2 retries

        return cls(openai.AsyncOpenAI(), settings.openai_model)

    async def generate(self, messages: list[dict[str, str]]) -> str:
        import openai

        try:
            response = await self._client.chat.completions.create(
                model=self.model_name, messages=messages, temperature=0
            )
        except openai.APITimeoutError as exc:
            raise RagDependencyError(
                "llm", str(exc), target=self.model_name, kind="timeout"
            ) from exc
        except openai.APIError as exc:
            raise RagDependencyError("llm", str(exc), target=self.model_name) from exc
        return response.choices[0].message.content or ""
