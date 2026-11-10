"""``RagPorts`` turns ``RagSettings`` into ready ports and owns what is behind them.

One instance serves one process. The workflow app asks for all four ports,
``rag-ingest`` for the embedder and the vector store, a model server for one
in-process model; nothing else is built, so no unused weights load and no unused
client is created. The model thread pool, the HTTP client and the bearer token
are created on first need, shared, and released when the ``async with`` block
ends; so is the thread budget, from the first CPU counter that answers
(``RAG_MODEL_CPUS``, the Domino tier, the container; see ``cpu_count.py``). A
port handed to the constructor is used as given and never closed here.

Open ports from one task (an app lifespan or a CLI ``main``): first requests are
not guarded against running concurrently. This is the only adapter module that
reads ``RagSettings``; the others take their values as constructor arguments.

Example::

    async with RagPorts(RagSettings()) as ports:
        deps = await ports.workflow_dependencies()
        workflow = RagWorkflow(deps, ports.settings.workflow_parameters())
        result = await workflow.run("How many vacation days?")
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
from ..domain.thread_budget import ThreadBudget
from .cpu_count import (
    ContainerCpus,
    CpuCounter,
    DominoHardwareTierCpus,
    FixedCpus,
    work_out_thread_budget,
)
from .elasticsearch_store import ElasticsearchVectorStore
from .fakes import FakeChatModel, FakeEmbedder, FakeReranker, InMemoryVectorStore
from .http_policy import HttpPolicy
from .local_models import (
    CrossEncoderReranker,
    NoReranker,
    SentenceTransformerEmbedder,
    model_executor,
)
from .native_threads import settle_torch_thread
from .openai_chat import OpenAIChatModel
from .ovms import BearerToken, OvmsEmbedder, OvmsReranker

T = TypeVar("T")
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
        container_cpus: CpuCounter | None = None,
    ) -> None:
        self.settings = settings
        self._container_cpus = container_cpus or ContainerCpus()
        self._supplied = _SuppliedPorts(embedder, vector_store, reranker, chat_model)
        self._http_policy = HttpPolicy(
            settings.http_timeout_s, settings.http_retries, settings.http_backoff_s
        )
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
                policy=self._http_policy,
                batch_size=settings.embedder_batch_size,
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
                policy=self._http_policy,
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
        settings = self.settings
        store = ElasticsearchVectorStore.open(
            settings.es_url, index=settings.es_index, api_key=settings.es_api_key
        )
        self._stack("vector_store").push_async_callback(store.aclose)
        return store

    async def _seeded_in_memory_store(self) -> VectorStorePort:
        store = InMemoryVectorStore()
        documents = synthetic_corpus(self.settings.fake_store_seed_documents)
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
        budget, pool = await self._thread_budget(), await self._model_pool()
        return SentenceTransformerEmbedder.from_pretrained(
            self.settings.embedder_model,
            pool,
            batch_size=self.settings.embedder_batch_size,
            threads=budget.embedder_threads,
        )

    async def _local_reranker(self) -> RerankerPort:
        return await self._once("local_reranker", "local", self._load_reranker)

    async def _load_reranker(self) -> RerankerPort:
        budget, pool = await self._thread_budget(), await self._model_pool()
        return CrossEncoderReranker.from_pretrained(
            self.settings.reranker_model,
            pool,
            batch_size=self.settings.reranker_batch_size,
            threads=budget.reranker_threads,
        )

    async def _model_pool(self) -> ThreadPoolExecutor:
        return await self._once("model_pool", OWNED, self._start_model_pool)

    async def _start_model_pool(self) -> ThreadPoolExecutor:
        workers = (await self._thread_budget()).workers
        # Only real local models reach here; see settle_torch_thread for why.
        pool = model_executor(workers, on_thread_start=settle_torch_thread)
        self._stack("model_pool").callback(pool.shutdown, wait=False)
        return pool

    async def _thread_budget(self) -> ThreadBudget:
        return await self._once("thread_budget", OWNED, self._work_out_budget)

    async def _work_out_budget(self) -> ThreadBudget:
        settings = self.settings
        counters: list[CpuCounter] = [FixedCpus(settings.model_cpus)]
        if settings.domino_api_proxy and settings.domino_hardware_tier_id:
            counters.append(
                DominoHardwareTierCpus(
                    await self._http_client(),
                    settings.domino_api_proxy,
                    settings.domino_hardware_tier_id,
                    self._http_policy,
                )
            )
        return await work_out_thread_budget(
            [*counters, self._container_cpus],
            max_workers=settings.model_workers,
            embedder_threads=settings.embedder_threads,
            reranker_threads=settings.reranker_threads,
        )

    async def _http_client(self) -> httpx.AsyncClient:
        return await self._once("http_client", OWNED, self._start_http_client)

    async def _start_http_client(self) -> httpx.AsyncClient:
        timeout = self._http_policy.timeout_s
        client = httpx.AsyncClient(timeout=timeout, transport=self._transport)
        self._stack("http_client").push_async_callback(client.aclose)
        return client

    async def _bearer_token(self) -> BearerToken:
        # One token cache for every model-server adapter: one Domino fetch, not two.
        return await self._once("bearer_token", OWNED, self._start_bearer_token)

    async def _start_bearer_token(self) -> BearerToken:
        settings = self.settings
        return BearerToken(
            await self._http_client(),
            settings.ovms_auth,
            token_url=settings.domino_token_url,
            ttl_s=settings.domino_token_ttl_s,
            policy=self._http_policy,
        )

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
