"""``CompositionRoot`` turns ``RagSettings`` into ready ports and owns what they use.

One instance serves one process. The workflow app asks for all four ports,
``rag-ingest`` for the embedder and the vector store, a model server for one
in-process model; nothing else is built, so no unused weights load and no unused
client is created. This package is the only one that reads ``RagSettings`` to
build an adapter; the adapters take their values as constructor arguments.

Each job has its own module, and this class only wires them together:

- ``resource_scope.py``: each thing is opened once and all are closed together;
- ``http_resources.py`` and ``local_model_resources.py``: what the
  implementations share (the HTTP client, the thread budget, the model pool);
- one factory per port (``embedder_factory.py``, ``reranker_factory.py``,
  ``vector_store_factory.py``, ``llm_factory.py``), which maps each Backend to
  what opens it: a new Backend is one method and one entry in one factory.

Open ports from one task (an app lifespan or a CLI ``main``): first requests are
not guarded against running concurrently.

Example:
    >>> async with CompositionRoot(RagSettings()) as composition:
    ...     deps = await composition.workflow_dependencies()
    ...     workflow = RagWorkflow(deps, composition.settings.workflow_parameters())
    ...     result = await workflow.run("How many vacation days?")
"""

from __future__ import annotations

from types import TracebackType
from typing import NamedTuple, TypedDict, Unpack

import httpx

from ..adapters.embedder.base_embedder import BaseEmbedder
from ..adapters.llm.base_llm import BaseLLM
from ..adapters.reranker.base_reranker import BaseReranker
from ..adapters.vector_store.base_vector_store import BaseLexicalSearch, BaseVectorStore
from ..application.rag_dependencies import RagDependencies
from ..config.rag_settings import RagSettings
from ..infrastructure.local_models.cpu_count import BaseCpuCounter, ContainerCpus
from .embedder_factory import EmbedderFactory
from .http_resources import HttpResources
from .llm_factory import LLMFactory
from .local_model_resources import LocalModelResources
from .reranker_factory import RerankerFactory
from .resource_scope import ResourceScope
from .vector_store_factory import VectorStoreFactory


class IngestPorts(NamedTuple):
    """The two ports ``rag-ingest`` needs."""

    embedder: BaseEmbedder
    vector_store: BaseVectorStore


class SuppliedPorts(TypedDict, total=False):
    """Ports handed in ready-made (test doubles, mostly), used as given."""

    embedder: BaseEmbedder | None
    vector_store: BaseVectorStore | None
    reranker: BaseReranker | None
    llm: BaseLLM | None


class CompositionRoot:
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
        >>> composition = CompositionRoot(settings, reranker=FailingReranker("x"))
        >>> async with composition:
        ...     embedder, store = await composition.ingest_ports()
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
        self._scope = scope = ResourceScope("CompositionRoot")
        http = HttpResources(settings, scope, transport)
        local_models = LocalModelResources(
            settings, scope, http, container_cpus or ContainerCpus()
        )
        self._embedders = EmbedderFactory(
            settings, scope, http, local_models, supplied.get("embedder")
        )
        self._rerankers = RerankerFactory(
            settings, scope, http, local_models, supplied.get("reranker")
        )
        self._llms = LLMFactory(settings, scope, http, supplied.get("llm"))
        self._vector_stores = VectorStoreFactory(
            settings, scope, self._embedders.open, supplied.get("vector_store")
        )

    async def __aenter__(self) -> CompositionRoot:
        await self._scope.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._scope.__aexit__(exc_type, exc, traceback)

    async def workflow_dependencies(self) -> RagDependencies:
        """Open all four ports, for the workflow app.

        The LLM opens first: it is cheap, so a bad generation setting fails
        before any weights load. If a port cannot open, the error propagates and
        nothing is returned.

        Returns:
            The embedder, vector store, reranker and LLM, and the store as
            lexical search when it ranks text (None otherwise).

        Example:
            >>> deps = await composition.workflow_dependencies()
        """
        llm = await self._llms.open()
        embedder = await self._embedders.open()
        store = await self._vector_stores.open()
        reranker = await self._rerankers.open()
        lexical = store if isinstance(store, BaseLexicalSearch) else None
        return RagDependencies(embedder, store, reranker, llm, lexical_search=lexical)

    async def ingest_ports(self) -> IngestPorts:
        """Open the embedder and the vector store, for ``rag-ingest``.

        The reranker and the LLM stay shut.

        Returns:
            The embedder and the vector store.

        Example:
            >>> embedder, store = await composition.ingest_ports()
        """
        return IngestPorts(
            await self._embedders.open(), await self._vector_stores.open()
        )

    async def local_embedder(self) -> BaseEmbedder:
        """Open the in-process embedder whatever ``embedding_backend`` says.

        A model server is what the ``api`` backend calls, so it always loads the
        local model.

        Returns:
            The supplied embedder if one was given, else the sentence-transformers
            one.

        Example:
            >>> embedder = await composition.local_embedder()
        """
        return await self._embedders.open_local()

    async def local_reranker(self) -> BaseReranker:
        """Open the in-process reranker whatever ``rerank_backend`` says.

        A model server is what the ``api`` backend calls, so it always loads the
        local model.

        Returns:
            The supplied reranker if one was given, else the cross-encoder one.

        Example:
            >>> reranker = await composition.local_reranker()
        """
        return await self._rerankers.open_local()
