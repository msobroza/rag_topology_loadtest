"""``RagPorts`` turns ``RagSettings`` into ready ports and owns what is behind them.

One instance serves one process. The workflow app asks for all four ports,
``rag-ingest`` for the embedder and the vector store, a model server for one
in-process model; nothing else is built, so no unused weights load and no unused
client is created. The model thread pool, the HTTP client and the bearer token
are created on first need, shared, and released when the ``async with`` block
ends. A port handed to the constructor is used as given and never closed here.

Open ports from one task (an app lifespan or a CLI ``main``): first requests are
not guarded against running concurrently.

Example::

    async with RagPorts(RagSettings()) as ports:
        deps = await ports.workflow_dependencies()
        result = await RagWorkflow(deps).run("How many vacation days?")
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass
from types import TracebackType
from typing import NamedTuple, TypeVar, cast

import httpx

from ..application.ingest import ingest
from ..config.settings import RagSettings
from ..domain.corpus import chunk, synthetic_corpus
from ..domain.errors import RagDependencyError, describe_error
from ..domain.ports import (
    ChatModelPort,
    EmbedderPort,
    RagDependencies,
    RerankerPort,
    VectorStorePort,
)
from .elasticsearch_store import ElasticsearchVectorStore
from .fakes import FakeChatModel, FakeEmbedder, FakeReranker, InMemoryVectorStore
from .local_models import (
    CrossEncoderReranker,
    NoReranker,
    SentenceTransformerEmbedder,
    model_executor,
)
from .openai_chat import OpenAIChatModel
from .ovms import TIMEOUT_S, BearerToken, OvmsEmbedder, OvmsReranker

T = TypeVar("T")
# Same size as ``make ingest`` so a fake-store app answers like an ingested one.
FAKE_STORE_SEED_DOCUMENTS = 200
OWNED = "in-process"  # "backend" of the resources RagPorts creates for itself


class IngestPorts(NamedTuple):
    """The two ports ``rag-ingest`` needs."""

    embedder: EmbedderPort
    vector_store: VectorStorePort


@dataclass(frozen=True)
class _SuppliedPorts:
    embedder: EmbedderPort | None
    vector_store: VectorStorePort | None
    reranker: RerankerPort | None
    chat_model: ChatModelPort | None


class RagPorts:
    """Ready ports for ``settings``; see the module docstring for the rules.

    Example::

        ports = RagPorts(settings, reranker=FailingReranker("timeout"))  # a test double
        async with ports:
            embedder, store = await ports.ingest_ports()
    """

    def __init__(
        self,
        settings: RagSettings,
        *,
        embedder: EmbedderPort | None = None,
        vector_store: VectorStorePort | None = None,
        reranker: RerankerPort | None = None,
        chat_model: ChatModelPort | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self._supplied = _SuppliedPorts(embedder, vector_store, reranker, chat_model)
        self._transport = transport
        self._opened: dict[str, object] = {}
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> RagPorts:
        if self._exit_stack is not None:
            raise RuntimeError("RagPorts is already open; exit it before re-entering")
        self._exit_stack = AsyncExitStack()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        stack, self._exit_stack = self._stack("__aexit__"), None
        self._opened.clear()
        await stack.aclose()

    # --- one entry point per caller ----------------------------------------------

    async def workflow_dependencies(self) -> RagDependencies:
        """All four ports, for the workflow app.

        The chat model opens first so a missing API key fails before any weights
        load. If a port cannot open, the error propagates and nothing is returned.
        """
        chat_model = await self._chat_model()
        embedder = await self._embedder()
        vector_store = await self._vector_store()
        reranker = await self._reranker()
        return RagDependencies(embedder, vector_store, reranker, chat_model)

    async def ingest_ports(self) -> IngestPorts:
        """The embedder and the vector store; the reranker and chat model stay shut."""
        return IngestPorts(await self._embedder(), await self._vector_store())

    async def local_embedder(self) -> EmbedderPort:
        """The in-process embedder whatever ``embedder_backend`` says: a model server
        is what the ``ovms`` backend calls, so it must never resolve to ``ovms``."""
        if self._supplied.embedder is not None:
            return self._supplied.embedder
        return await self._local_embedder()

    async def local_reranker(self) -> RerankerPort:
        """The in-process reranker whatever ``reranker_backend`` says."""
        if self._supplied.reranker is not None:
            return self._supplied.reranker
        return await self._local_reranker()

    # --- ports, by backend ----------------------------------------------------------

    async def _embedder(self) -> EmbedderPort:
        if self._supplied.embedder is not None:
            return self._supplied.embedder
        backend = self.settings.embedder_backend
        return await self._once("embedder", backend, self._open_embedder)

    async def _open_embedder(self) -> EmbedderPort:
        settings = self.settings
        if settings.embedder_backend == "fake":
            return FakeEmbedder()
        if settings.embedder_backend == "ovms":
            return OvmsEmbedder(
                await self._http_client(),
                settings.ovms_embeddings_url,
                settings.ovms_embeddings_model,
                token=await self._bearer_token(),
            )
        return await self._local_embedder()

    async def _reranker(self) -> RerankerPort:
        if self._supplied.reranker is not None:
            return self._supplied.reranker
        backend = self.settings.reranker_backend
        return await self._once("reranker", backend, self._open_reranker)

    async def _open_reranker(self) -> RerankerPort:
        settings = self.settings
        if settings.reranker_backend == "none":
            return NoReranker()
        if settings.reranker_backend == "fake":
            return FakeReranker()
        if settings.reranker_backend == "ovms":
            return OvmsReranker(
                await self._http_client(),
                settings.ovms_rerank_url,
                settings.ovms_rerank_model,
                token=await self._bearer_token(),
            )
        return await self._local_reranker()

    async def _vector_store(self) -> VectorStorePort:
        if self._supplied.vector_store is not None:
            return self._supplied.vector_store
        backend = self.settings.vector_store_backend
        return await self._once("vector_store", backend, self._open_vector_store)

    async def _open_vector_store(self) -> VectorStorePort:
        if self.settings.vector_store_backend == "fake":
            return await self._seeded_in_memory_store()
        store = ElasticsearchVectorStore.open(
            self.settings.es_url, api_key=self.settings.es_api_key
        )
        self._stack("vector_store").push_async_callback(store.aclose)
        return store

    async def _seeded_in_memory_store(self) -> VectorStorePort:
        store = InMemoryVectorStore()
        documents = synthetic_corpus(FAKE_STORE_SEED_DOCUMENTS)
        passages = [passage for document in documents for passage in chunk(document)]
        await ingest(passages, await self._embedder(), store)
        return store

    async def _chat_model(self) -> ChatModelPort:
        if self._supplied.chat_model is not None:
            return self._supplied.chat_model
        backend = self.settings.llm_backend
        return await self._once("chat_model", backend, self._open_chat_model)

    async def _open_chat_model(self) -> ChatModelPort:
        if self.settings.llm_backend == "fake":
            return FakeChatModel()
        chat_model = OpenAIChatModel.open(self.settings.openai_model)
        self._stack("chat_model").push_async_callback(chat_model.aclose)
        return chat_model

    # --- in-process models and the resources they share -----------------------------

    async def _local_embedder(self) -> EmbedderPort:
        return await self._once("local_embedder", "local", self._load_embedder)

    async def _load_embedder(self) -> EmbedderPort:
        pool = await self._model_pool()
        return SentenceTransformerEmbedder.from_pretrained(
            self.settings.embedder_model, pool
        )

    async def _local_reranker(self) -> RerankerPort:
        return await self._once("local_reranker", "local", self._load_reranker)

    async def _load_reranker(self) -> RerankerPort:
        pool = await self._model_pool()
        return CrossEncoderReranker.from_pretrained(self.settings.reranker_model, pool)

    async def _model_pool(self) -> ThreadPoolExecutor:
        return await self._once("model_pool", OWNED, self._start_model_pool)

    async def _start_model_pool(self) -> ThreadPoolExecutor:
        pool = model_executor()
        self._stack("model_pool").callback(pool.shutdown, wait=False)
        return pool

    async def _http_client(self) -> httpx.AsyncClient:
        return await self._once("http_client", OWNED, self._start_http_client)

    async def _start_http_client(self) -> httpx.AsyncClient:
        client = httpx.AsyncClient(timeout=TIMEOUT_S, transport=self._transport)
        self._stack("http_client").push_async_callback(client.aclose)
        return client

    async def _bearer_token(self) -> BearerToken:
        # One token cache for every model-server adapter: one Domino fetch, not two.
        return await self._once("bearer_token", OWNED, self._start_bearer_token)

    async def _start_bearer_token(self) -> BearerToken:
        return BearerToken(await self._http_client(), self.settings.ovms_auth)

    # --- bookkeeping ----------------------------------------------------------------

    async def _once(
        self, name: str, backend: str, open_it: Callable[[], Awaitable[T]]
    ) -> T:
        """Open ``name`` on first request and reuse it until the block ends."""
        self._stack(name)
        if name not in self._opened:
            try:
                self._opened[name] = await open_it()
            except RagDependencyError:
                raise  # a port this one depends on already says what failed
            except Exception as exc:
                detail = f"could not open backend {backend!r}: {describe_error(exc)}"
                raise RagDependencyError(name, detail) from exc
        return cast(T, self._opened[name])

    def _stack(self, requested: str) -> AsyncExitStack:
        if self._exit_stack is None:
            raise RuntimeError(
                f"RagPorts: {requested!r} requested outside 'async with RagPorts(...)'"
            )
        return self._exit_stack
