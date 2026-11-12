"""``RagPorts`` turns ``RagSettings`` into ready ports and owns what is behind them.

One instance serves one process. The workflow app asks for all four ports,
``rag-ingest`` for the embedder and the vector store, a model server for one
in-process model; nothing else is built, so no unused weights load and no unused
client is created. The model thread pool and the HTTP client are created on
first need, shared, and released when the ``async with`` block ends; so is the
thread budget, from the first CPU counter that answers
(``RAG_MODEL_CPUS``, the Domino tier, the container; see
``local_models/cpu_count.py``). A port handed to the constructor is used as
given and never closed here. Each component maps its backend names to the
method that opens it, so a new backend is one method and one entry.

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
from ..config.rag_settings import RagSettings
from ..domain.corpus import chunk, synthetic_corpus
from ..domain.errors import RagDependencyError, describe_error
from ..domain.ports import (
    BaseEmbedder,
    BaseLexicalSearch,
    BaseLLM,
    BaseReranker,
    BaseVectorStore,
    RagDependencies,
)
from ..domain.thread_budget import ThreadBudget
from .fakes.fake_models import FakeEmbedder, FakeLLM, FakeReranker
from .fakes.in_memory_vector_store import InMemoryVectorStore
from .local_models.cpu_count import (
    BaseCpuCounter,
    ContainerCpus,
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
from .model_api.endpoint_adapters import (
    RunTokenSource,
    open_api_embedder,
    open_api_llm,
    open_api_reranker,
)
from .shared.http_policy import HttpPolicy
from .vector_store.chroma_store import ChromaVectorStore
from .vector_store.elasticsearch_store import ElasticsearchVectorStore

T = TypeVar("T")
Opener = Callable[[], Awaitable[T]]  # opens one port for one backend name
OWNED = "in-process"  # "backend" of the resources RagPorts creates for itself


class IngestPorts(NamedTuple):
    """The two ports ``rag-ingest`` needs."""

    embedder: BaseEmbedder
    vector_store: BaseVectorStore


class SuppliedPorts(TypedDict, total=False):
    """Ports handed to ``RagPorts`` ready-made (test doubles, mostly), used as given."""

    embedder: BaseEmbedder | None
    vector_store: BaseVectorStore | None
    reranker: BaseReranker | None
    llm: BaseLLM | None


class RagPorts:
    """Ready ports for ``settings``; see the module docstring for the rules.

    Args:
        settings: The backends, URLs and sizes the ports are built from.
        transport: The HTTP transport of the shared client; None for the real
            network (tests pass an ``httpx.MockTransport``).
        container_cpus: Counts the container's CPUs when nothing earlier in the
            chain answers; None for ``ContainerCpus()``.
        **supplied: Ready-made ports by name (``embedder``, ``vector_store``,
            ``reranker``, ``llm``), used as given and never closed here.

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
        container_cpus: BaseCpuCounter | None = None,
        **supplied: Unpack[SuppliedPorts],
    ) -> None:
        self.settings = settings
        self._container_cpus = container_cpus or ContainerCpus()
        self._supplied = supplied
        self._http_policy = HttpPolicy(
            settings.http_timeout_s, settings.http_retries, settings.http_backoff_s
        )
        self._run_tokens = RunTokenSource(
            settings.domino_token_url, settings.domino_token_ttl_s
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

        The LLM opens first: it is cheap, so a bad generation setting fails
        before any weights load. If a port cannot open, the error propagates and
        nothing is returned.

        Returns:
            The embedder, vector store, reranker and LLM, and the store as
            lexical search when it ranks text (None otherwise).
        """
        llm = await self._llm()
        embedder = await self._embedder()
        store = await self._vector_store()
        reranker = await self._reranker()
        lexical = store if isinstance(store, BaseLexicalSearch) else None
        return RagDependencies(embedder, store, reranker, llm, lexical_search=lexical)

    async def ingest_ports(self) -> IngestPorts:
        """Open the embedder and the vector store, for ``rag-ingest``.

        The reranker and the LLM stay shut.

        Returns:
            The embedder and the vector store.
        """
        return IngestPorts(await self._embedder(), await self._vector_store())

    async def local_embedder(self) -> BaseEmbedder:
        """Open the in-process embedder whatever ``embedding_backend`` says.

        A model server is what the ``api`` backend calls, so it always loads the
        local model.

        Returns:
            The supplied embedder if one was given, else the sentence-transformers
            one.
        """
        if (supplied := self._supplied.get("embedder")) is not None:
            return supplied
        return await self._local_embedder()

    async def local_reranker(self) -> BaseReranker:
        """Open the in-process reranker whatever ``rerank_backend`` says.

        A model server is what the ``api`` backend calls, so it always loads the
        local model.

        Returns:
            The supplied reranker if one was given, else the cross-encoder one.
        """
        if (supplied := self._supplied.get("reranker")) is not None:
            return supplied
        return await self._local_reranker()

    # --- ports, by backend: one opener per backend name ------------------------------

    async def _embedder(self) -> BaseEmbedder:
        if (supplied := self._supplied.get("embedder")) is not None:
            return supplied
        backend = self.settings.embedding_backend
        return await self._once("embedder", backend, self._open_embedder)

    async def _open_embedder(self) -> BaseEmbedder:
        openers: dict[str, Opener[BaseEmbedder]] = {
            "local": self._local_embedder,
            "api": self._api_embedder,
            "fake": lambda: _built(FakeEmbedder()),
        }
        return await openers[self.settings.embedding_backend]()

    async def _api_embedder(self) -> BaseEmbedder:
        return open_api_embedder(
            await self._http_client(),
            self.settings.embedding_api(),
            tokens=self._run_tokens,
            policy=self._http_policy,
            batch_size=self.settings.embedding_batch_size,
        )

    async def _reranker(self) -> BaseReranker:
        if (supplied := self._supplied.get("reranker")) is not None:
            return supplied
        backend = self.settings.rerank_backend
        return await self._once("reranker", backend, self._open_reranker)

    async def _open_reranker(self) -> BaseReranker:
        openers: dict[str, Opener[BaseReranker]] = {
            "local": self._local_reranker,
            "api": self._api_reranker,
            "fake": lambda: _built(FakeReranker()),
            "none": lambda: _built(NoReranker()),
        }
        return await openers[self.settings.rerank_backend]()

    async def _api_reranker(self) -> BaseReranker:
        return open_api_reranker(
            await self._http_client(),
            self.settings.rerank_api(),
            tokens=self._run_tokens,
            policy=self._http_policy,
        )

    async def _vector_store(self) -> BaseVectorStore:
        if (supplied := self._supplied.get("vector_store")) is not None:
            return supplied
        backend = self.settings.vector_store_backend
        return await self._once("vector_store", backend, self._open_vector_store)

    async def _open_vector_store(self) -> BaseVectorStore:
        openers: dict[str, Opener[BaseVectorStore]] = {
            "elasticsearch": self._elasticsearch_store,
            "chroma": self._chroma_store,
            "fake": self._seeded_in_memory_store,
        }
        return await openers[self.settings.vector_store_backend]()

    async def _elasticsearch_store(self) -> BaseVectorStore:
        settings = self.settings
        store = ElasticsearchVectorStore.open(
            settings.es_url,
            index=settings.es_index,
            api_key=settings.es_api_key,
            text_analyzer=settings.es_text_analyzer,
        )
        self._stack("vector_store").push_async_callback(store.aclose)
        return store

    async def _chroma_store(self) -> BaseVectorStore:
        settings = self.settings
        return await ChromaVectorStore.open(
            url=settings.chroma_url,
            path=settings.chroma_path,
            collection=settings.chroma_collection,
        )

    async def _seeded_in_memory_store(self) -> BaseVectorStore:
        store = InMemoryVectorStore()
        documents = synthetic_corpus(self.settings.fake_store_seed_documents)
        passages = [passage for document in documents for passage in chunk(document)]
        await ingest(passages, await self._embedder(), store)
        return store

    async def _llm(self) -> BaseLLM:
        if (supplied := self._supplied.get("llm")) is not None:
            return supplied
        backend = self.settings.generation_backend
        return await self._once("llm", backend, self._open_llm)

    async def _open_llm(self) -> BaseLLM:
        openers: dict[str, Opener[BaseLLM]] = {
            "api": self._api_llm,
            "fake": lambda: _built(FakeLLM()),
        }
        return await openers[self.settings.generation_backend]()

    async def _api_llm(self) -> BaseLLM:
        settings = self.settings
        # An answer takes longer than a model-server call: its own timeout, the
        # shared retries and backoff.
        policy = HttpPolicy(
            settings.generation_timeout_s,
            settings.http_retries,
            settings.http_backoff_s,
        )
        return open_api_llm(
            await self._http_client(),
            settings.generation_api(),
            tokens=self._run_tokens,
            policy=policy,
            temperature=settings.generation_temperature,
        )

    # --- in-process models and the resources they share -----------------------------

    async def _local_embedder(self) -> BaseEmbedder:
        return await self._once("local_embedder", "local", self._load_embedder)

    async def _load_embedder(self) -> BaseEmbedder:
        budget, pool = await self._thread_budget(), await self._model_pool()
        model_id = self.settings.embedding_model
        return SentenceTransformerEmbedder.from_pretrained(
            await self._local_model(model_id, export_embedder, pool),
            pool,
            batch_size=self.settings.embedding_batch_size,
            threads=budget.embedder_threads,
        )

    async def _local_reranker(self) -> BaseReranker:
        return await self._once("local_reranker", "local", self._load_reranker)

    async def _load_reranker(self) -> BaseReranker:
        budget, pool = await self._thread_budget(), await self._model_pool()
        model_id = self.settings.rerank_model
        return CrossEncoderReranker.from_pretrained(
            await self._local_model(model_id, export_reranker, pool),
            pool,
            batch_size=self.settings.rerank_batch_size,
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
        counters: list[BaseCpuCounter] = [FixedCpus(settings.model_cpus)]
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
            embedder_threads=settings.embedding_threads,
            reranker_threads=settings.rerank_threads,
        )

    async def _http_client(self) -> httpx.AsyncClient:
        return await self._once("http_client", OWNED, self._start_http_client)

    async def _start_http_client(self) -> httpx.AsyncClient:
        timeout = self._http_policy.timeout_s
        client = httpx.AsyncClient(timeout=timeout, transport=self._transport)
        self._stack("http_client").push_async_callback(client.aclose)
        return client

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


async def _built(port: T) -> T:
    # The opener of a port that needs nothing opened: it is built on the spot.
    return port
