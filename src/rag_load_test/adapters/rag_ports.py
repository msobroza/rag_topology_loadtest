"""``RagPorts`` turns ``RagSettings`` into ready ports and owns what is behind them.

One instance serves one process. The workflow app asks for all four ports,
``rag-ingest`` for the embedder and the vector store, a model server for one
in-process model; nothing else is built, so no unused weights load and no unused
client is created. The model thread pool, the HTTP client and the bearer token
are created on first need, shared, and released when the ``async with`` block
ends; so is the thread budget, from the first CPU counter that answers
(``RAG_MODEL_CPUS``, the Domino tier, the container; see
``local_models/cpu_count.py``). A port handed to the constructor is used as
given and never closed here.

Open ports from one task (an app lifespan or a CLI ``main``): first requests are
not guarded against running concurrently. This is the only adapter module that
reads ``RagSettings``; the others take their values as constructor arguments.

Example:
    >>> async with RagPorts(RagSettings()) as ports:
    ...     deps = await ports.workflow_dependencies()
    ...     workflow = RagWorkflow(deps, ports.settings.workflow_parameters())
    ...     result = await workflow.run("How many vacation days?")
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import AsyncExitStack
from types import TracebackType
from typing import NamedTuple, TypedDict, TypeVar, Unpack, cast

import httpx

from ..application.ingest import ingest
from ..config.settings import RagSettings
from ..domain.bm25 import Bm25Index
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
from .chat_model.openai_chat import OpenAIChatModel
from .fakes.fake_models import FakeChatModel, FakeEmbedder, FakeReranker
from .fakes.in_memory_vector_store import InMemoryVectorStore
from .hosted_models.openai_embedder import OpenAIEmbedder
from .hosted_models.rerank_api import RerankApiReranker
from .local_models.cpu_count import (
    ContainerCpus,
    CpuCounter,
    DominoHardwareTierCpus,
    FixedCpus,
    work_out_thread_budget,
)
from .local_models.local_model import LocalModel
from .local_models.model_provisioning import Exporter, provision_local_model
from .local_models.native_threads import settle_torch_thread
from .local_models.openvino_export import export_embedder, export_reranker
from .local_models.sentence_transformer_models import (
    CrossEncoderReranker,
    NoReranker,
    SentenceTransformerEmbedder,
    model_executor,
)
from .ovms.ovms_adapters import BearerToken, OvmsEmbedder, OvmsReranker
from .shared.http_policy import HttpPolicy
from .vector_store.chroma_store import ChromaVectorStore
from .vector_store.elasticsearch_store import ElasticsearchVectorStore

T = TypeVar("T")
OWNED = "in-process"  # "backend" of the resources RagPorts creates for itself


class IngestPorts(NamedTuple):
    """The two ports ``rag-ingest`` needs."""

    embedder: EmbedderPort
    vector_store: VectorStorePort


class SuppliedPorts(TypedDict, total=False):
    """Ports handed to ``RagPorts`` ready-made (test doubles, mostly), used as given."""

    embedder: EmbedderPort | None
    vector_store: VectorStorePort | None
    reranker: RerankerPort | None
    chat_model: ChatModelPort | None


class RagPorts:
    """Ready ports for ``settings``; see the module docstring for the rules.

    Args:
        settings: The backends, URLs and sizes the ports are built from.
        transport: The HTTP transport of the shared client; None for the real
            network (tests pass an ``httpx.MockTransport``).
        container_cpus: Counts the container's CPUs when nothing earlier in the
            chain answers; None for ``ContainerCpus()``.
        **supplied: Ready-made ports by name (``embedder``, ``vector_store``,
            ``reranker``, ``chat_model``), used as given and never closed here.

    Example:
        >>> ports = RagPorts(settings, reranker=FailingReranker("timeout"))
        >>> async with ports:
        ...     embedder, store = await ports.ingest_ports()
    """

    def __init__(
        self,
        settings: RagSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        container_cpus: CpuCounter | None = None,
        **supplied: Unpack[SuppliedPorts],
    ) -> None:
        self.settings = settings
        self._container_cpus = container_cpus or ContainerCpus()
        self._supplied = supplied
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
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        stack, self._exit_stack = self._stack("__aexit__"), None
        self._opened.clear()
        await stack.aclose()

    # --- one entry point per caller ----------------------------------------------

    async def workflow_dependencies(self) -> RagDependencies:
        """Open all four ports, for the workflow app.

        The chat model opens first so a missing API key fails before any weights
        load. If a port cannot open, the error propagates and nothing is returned.

        Returns:
            The embedder, vector store, reranker and chat model.
        """
        chat_model = await self._chat_model()
        embedder = await self._embedder()
        vector_store = await self._vector_store()
        reranker = await self._reranker()
        return RagDependencies(embedder, vector_store, reranker, chat_model)

    async def ingest_ports(self) -> IngestPorts:
        """Open the embedder and the vector store, for ``rag-ingest``.

        The reranker and chat model stay shut.

        Returns:
            The embedder and the vector store.
        """
        return IngestPorts(await self._embedder(), await self._vector_store())

    async def local_embedder(self) -> EmbedderPort:
        """Open the in-process embedder whatever ``embedder_backend`` says.

        A model server is what the ``ovms`` backend calls, so it must never resolve
        to ``ovms``.

        Returns:
            The supplied embedder if one was given, else the sentence-transformers
            one.
        """
        if (supplied := self._supplied.get("embedder")) is not None:
            return supplied
        return await self._local_embedder()

    async def local_reranker(self) -> RerankerPort:
        """Open the in-process reranker whatever ``reranker_backend`` says.

        Returns:
            The supplied reranker if one was given, else the cross-encoder one.
        """
        if (supplied := self._supplied.get("reranker")) is not None:
            return supplied
        return await self._local_reranker()

    # --- ports, by backend ----------------------------------------------------------

    async def _embedder(self) -> EmbedderPort:
        if (supplied := self._supplied.get("embedder")) is not None:
            return supplied
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
        if settings.embedder_backend == "openai":
            return await self._open_openai_embedder()
        return await self._local_embedder()

    async def _open_openai_embedder(self) -> EmbedderPort:
        settings = self.settings
        embedder = OpenAIEmbedder.open(
            settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            batch_size=settings.embedder_batch_size,
            policy=self._http_policy,
        )
        self._stack("embedder").push_async_callback(embedder.aclose)
        return embedder

    async def _reranker(self) -> RerankerPort:
        if (supplied := self._supplied.get("reranker")) is not None:
            return supplied
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
        if settings.reranker_backend in ("cohere", "openai"):
            api = settings.rerank_api()
            return RerankApiReranker(
                await self._http_client(),
                api.url,
                api.model,
                api_key=api.api_key,
                policy=self._http_policy,
            )
        return await self._local_reranker()

    async def _vector_store(self) -> VectorStorePort:
        if (supplied := self._supplied.get("vector_store")) is not None:
            return supplied
        backend = self.settings.vector_store_backend
        return await self._once("vector_store", backend, self._open_vector_store)

    async def _open_vector_store(self) -> VectorStorePort:
        settings = self.settings
        if settings.vector_store_backend == "fake":
            return await self._seeded_in_memory_store()
        if settings.vector_store_backend == "chroma":
            return await ChromaVectorStore.open(
                url=settings.chroma_url,
                path=settings.chroma_path,
                collection=settings.chroma_collection,
                lexical_index=self._bm25_index(),
            )
        store = ElasticsearchVectorStore.open(
            settings.es_url,
            index=settings.es_index,
            api_key=settings.es_api_key,
            text_analyzer=settings.es_text_analyzer,
        )
        self._stack("vector_store").push_async_callback(store.aclose)
        return store

    # A fresh BM25 index for a store that has no ranked text search of its own.
    def _bm25_index(self) -> Bm25Index:
        return Bm25Index(k1=self.settings.bm25_k1, b=self.settings.bm25_b)

    async def _seeded_in_memory_store(self) -> VectorStorePort:
        store = InMemoryVectorStore(self._bm25_index())
        documents = synthetic_corpus(self.settings.fake_store_seed_documents)
        passages = [passage for document in documents for passage in chunk(document)]
        await ingest(passages, await self._embedder(), store)
        return store

    async def _chat_model(self) -> ChatModelPort:
        if (supplied := self._supplied.get("chat_model")) is not None:
            return supplied
        backend = self.settings.llm_backend
        return await self._once("chat_model", backend, self._open_chat_model)

    async def _open_chat_model(self) -> ChatModelPort:
        if self.settings.llm_backend == "fake":
            return FakeChatModel()
        settings = self.settings
        chat_model = OpenAIChatModel.open(
            settings.openai_model, temperature=settings.llm_temperature
        )
        self._stack("chat_model").push_async_callback(chat_model.aclose)
        return chat_model

    # --- in-process models and the resources they share -----------------------------

    async def _local_embedder(self) -> EmbedderPort:
        return await self._once("local_embedder", "local", self._load_embedder)

    async def _load_embedder(self) -> EmbedderPort:
        budget, pool = await self._thread_budget(), await self._model_pool()
        model_id = self.settings.embedder_model
        return SentenceTransformerEmbedder.from_pretrained(
            await self._local_model(model_id, export_embedder, pool),
            pool,
            batch_size=self.settings.embedder_batch_size,
            threads=budget.embedder_threads,
        )

    async def _local_reranker(self) -> RerankerPort:
        return await self._once("local_reranker", "local", self._load_reranker)

    async def _load_reranker(self) -> RerankerPort:
        budget, pool = await self._thread_budget(), await self._model_pool()
        model_id = self.settings.reranker_model
        return CrossEncoderReranker.from_pretrained(
            await self._local_model(model_id, export_reranker, pool),
            pool,
            batch_size=self.settings.reranker_batch_size,
            threads=budget.reranker_threads,
        )

    # Under openvino, a model whose folder is missing is exported there first (on
    # the model pool: it takes minutes); see local_models/model_provisioning.py.
    async def _local_model(
        self, model_id: str, export: Exporter, pool: Executor
    ) -> LocalModel:
        settings = self.settings
        return await provision_local_model(
            model_id,
            runtime=settings.local_model_runtime,
            models_dir=settings.openvino_models_dir,
            export=export,
            executor=pool,
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

    # Open ``name`` on first request and reuse it until the block ends; any
    # failure but a RagDependencyError is wrapped in one naming ``name``.
    async def _once(
        self, name: str, backend: str, open_it: Callable[[], Awaitable[T]]
    ) -> T:
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
