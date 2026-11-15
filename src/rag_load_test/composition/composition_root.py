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
- ``component_factory.py``: opens a component with the opener registered for
  its type and backend, one module each in ``openers/``: a new backend is one
  new module there, and no factory is edited.

The openers take each component's settings in the workflow file's shape,
derived here from the flat settings (``config/component_settings.py``).

Open ports from one task (an app lifespan or a CLI ``main``): first requests are
not guarded against running concurrently.

Example:
    >>> async with CompositionRoot(RagSettings()) as composition:
    ...     deps = await composition.workflow_dependencies()
    ...     graph = composition.settings.graph  # from the workflow file
    ...     workflow = build_workflow(graph, deps.components())
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
from ..config.component_settings import component_settings_of
from ..config.rag_settings import RagSettings
from ..domain.roles import ModelServerPart
from ..infrastructure.local_models.cpu_count import BaseCpuCounter, ContainerCpus
from .calibrated_stages import ComponentCalibration
from .component_factory import ComponentFactory
from .http_resources import HttpResources
from .local_model_resources import LocalModelResources
from .openers.component_openers import (
    EMBEDDER_OPENERS,
    LLM_OPENERS,
    RERANKER_OPENERS,
    VECTOR_STORE_OPENERS,
    SharedResources,
)
from .openers.embedder_on_local import calibrate_embedder_on_local
from .openers.reranker_on_local import calibrate_reranker_on_local
from .resource_scope import ResourceScope


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
        self._components = component_settings_of(settings)
        self._scope = scope = ResourceScope("CompositionRoot")
        http = HttpResources(settings, scope, transport)
        self._local_models = local_models = LocalModelResources(
            settings, scope, http, container_cpus or ContainerCpus()
        )
        shared = SharedResources(scope, http, local_models, self._embedder)
        self._factory = ComponentFactory(shared, supplied)

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
        components, factory = self._components, self._factory
        llm = await factory.open(LLM_OPENERS, components.llm)
        embedder = await self._embedder()
        store = await self._vector_store()
        reranker = await factory.open(RERANKER_OPENERS, components.reranker)
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
        return IngestPorts(await self._embedder(), await self._vector_store())

    async def local_embedder(self) -> BaseEmbedder:
        """Open the in-process embedder whatever its backend says.

        A model server is what the ``api`` backend calls, so it always loads the
        local model.

        Returns:
            The supplied embedder if one was given, else the sentence-transformers
            one.

        Example:
            >>> embedder = await composition.local_embedder()
        """
        embedder = self._components.embedder
        return await self._factory.open_local(EMBEDDER_OPENERS, embedder)

    async def local_reranker(self) -> BaseReranker:
        """Open the in-process reranker whatever its backend says.

        A model server is what the ``api`` backend calls, so it always loads the
        local model.

        Returns:
            The supplied reranker if one was given, else the cross-encoder one.

        Example:
            >>> reranker = await composition.local_reranker()
        """
        reranker = self._components.reranker
        return await self._factory.open_local(RERANKER_OPENERS, reranker)

    async def calibration(self, part: ModelServerPart) -> ComponentCalibration:
        """Calibrate one component's in-process model on this machine, on demand.

        The model loads as an App would load it, whatever the backend says, and
        every parameter calibration chooses is searched, pinned or not, under
        the settings' budget and rounds. A ready-made port is not timed.

        Args:
            part: ``embedder`` or ``reranker``.

        Returns:
            The report, the CPUs and runtime it ran on, and the settings that
            pin the choice.

        Example:
            >>> calibration = await composition.calibration("reranker")
            >>> calibration.pinned
            {'RAG_RERANK_THREADS': 2, 'RAG_RERANK_BATCH_SIZE': 2}
        """
        components, local_models = self._components, self._local_models
        if part == "embedder":
            return await calibrate_embedder_on_local(components.embedder, local_models)
        return await calibrate_reranker_on_local(components.reranker, local_models)

    async def _embedder(self) -> BaseEmbedder:
        # Also what seeds a fake store, so its vectors are the ones queries get.
        return await self._factory.open(EMBEDDER_OPENERS, self._components.embedder)

    async def _vector_store(self) -> BaseVectorStore:
        store = self._components.vector_store
        return await self._factory.open(VECTOR_STORE_OPENERS, store)
