"""``CompositionRoot`` makes ready ports of ``WorkflowSettings`` and owns what they use.

One instance serves one process. A command opens what it needs, by component
name: the workflow app the components its graph's stages call, ``qp-ingest``
an embedder and the vector store, a model server and ``qp-calibrate`` the one
component they are named; nothing else is built, so no unused weights load and
no unused client is created. This package is the only one that reads
``WorkflowSettings`` to build an adapter; the adapters take their values as
constructor arguments.

Each job has its own module, and this class only wires them together:

- ``resource_scope.py``: each thing is opened once and all are closed together;
- ``http_resources.py`` and ``local_model_resources.py``: what the
  implementations share (the HTTP client, the thread budget, the model pool);
- ``component_factory.py``: opens a component with the opener registered for
  its type and backend, one module each in ``openers/``: a new backend is one
  new module there, and no factory is edited;
- ``registered_components.py``: opens a component of a type that registers
  its implementations on their class line, an extension's, by its own class.

The openers take each component's settings in the workflow file's shape
(``config/component_settings.py``).

Open ports from one task (an app lifespan or a CLI ``main``): first requests are
not guarded against running concurrently.

Example:
    >>> async with CompositionRoot(WorkflowSettings()) as composition:
    ...     deps = await composition.workflow_dependencies()
    ...     graph = composition.settings.graph  # from the workflow file
    ...     workflow = build_workflow(graph, deps.components())
    ...     result = await workflow.run("How many vacation days?")
"""

from __future__ import annotations

from types import TracebackType
from typing import NamedTuple, TypeVar

import httpx

from ..adapters.embedder.base_embedder import BaseEmbedder
from ..adapters.reranker.base_reranker import BaseReranker
from ..adapters.vector_store.base_vector_store import BaseVectorStore
from ..application.rag_dependencies import RagDependencies
from ..config.component_settings import (
    ComponentSettings,
    EmbedderSettings,
    RerankerSettings,
    VectorStoreSettings,
)
from ..config.registered_component_settings import RegisteredComponentSettings
from ..config.workflow_settings import WorkflowSettings
from ..domain.component_resources import ComponentResources
from ..infrastructure.local_models.cpu_count import BaseCpuCounter, ContainerCpus
from .calibrated_stages import ComponentCalibration
from .component_factory import ComponentFactory
from .http_resources import HttpResources
from .local_model_resources import LocalModelResources
from .openers.component_openers import (
    EMBEDDER_OPENERS,
    OPENERS_BY_TYPE,
    RERANKER_OPENERS,
    VECTOR_STORE_OPENERS,
    SharedResources,
)
from .openers.embedder_on_local import calibrate_embedder_on_local
from .openers.reranker_on_local import calibrate_reranker_on_local
from .resource_scope import ResourceScope

C = TypeVar("C", bound=ComponentSettings)


class IngestPorts(NamedTuple):
    """The two ports ``qp-ingest`` needs."""

    embedder: BaseEmbedder
    vector_store: BaseVectorStore


class CompositionRoot:
    """Ready ports for ``settings``; see the module docstring for the rules.

    Args:
        settings: The components, backends and sizes the ports are built from.
        transport: The HTTP transport of the shared client; None for the real
            network (tests pass an ``httpx.MockTransport``).
        on_domino: Whether the process runs as a Domino App, where a component
            on the ``served`` backend is called with the run token
            (``qp-serve --domino``); off Domino it is called with no key.
        container_cpus: Counts the container's CPUs when nothing earlier in the
            chain answers; None for ``ContainerCpus()``.
        **supplied: Ready-made ports by component name (``embedder``,
            ``vector_store``, ...), used as given and never closed here.

    Example:
        >>> composition = CompositionRoot(settings, reranker=FailingReranker("x"))
        >>> async with composition:
        ...     embedder, store = await composition.ingest_ports()
    """

    def __init__(
        self,
        settings: WorkflowSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        on_domino: bool = False,
        container_cpus: BaseCpuCounter | None = None,
        **supplied: object,
    ) -> None:
        self.settings = settings
        self._components = settings.components
        self._scope = scope = ResourceScope("CompositionRoot")
        http = HttpResources(settings, scope, transport, on_domino=on_domino)
        self._local_models = local_models = LocalModelResources(
            settings, scope, http, container_cpus or ContainerCpus()
        )
        shared = SharedResources(scope, http, local_models, self._store_embedder)
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
        """Open the components the graph's stages call, for the workflow app.

        An LLM opens first: it is cheap, so a bad generation setting fails
        before any weights load. The others open in the order the stages call
        them. If a component cannot open, the error propagates and nothing is
        returned.

        Returns:
            Each component a stage calls, open, by name.

        Raises:
            ValueError: If a stage calls a component that is not declared.

        Example:
            >>> deps = await composition.workflow_dependencies()
        """
        called = dict.fromkeys(s.call for s in self.settings.graph.stages.values())
        llms_first = sorted(
            called, key=lambda name: self._component(name).type != "llm"
        )
        return RagDependencies({name: await self._open(name) for name in llms_first})

    async def ingest_ports(self, embedder: str | None = None) -> IngestPorts:
        """Open an embedder and the vector store, for ``qp-ingest``.

        Nothing else opens. The store is the one the workflow file declares.

        Args:
            embedder: The embedder component to embed with; None for the only
                one the workflow file declares.

        Returns:
            The embedder and the vector store.

        Raises:
            ValueError: If ``embedder`` names no embedder, or none is named and
                the file declares several; or the file declares no vector
                store, or several.

        Example:
            >>> embedder, store = await composition.ingest_ports()
        """
        name = self.settings.ingestion_embedder(embedder)
        return IngestPorts(await self._embedder(name), await self._vector_store())

    async def local_embedder(self, name: str) -> BaseEmbedder:
        """Open the named embedder in this process, whatever its backend says.

        A model server is what the ``served`` backend calls, so it always loads the
        local model.

        Args:
            name: An embedder component of the workflow file.

        Returns:
            The supplied embedder if one was given, else the sentence-transformers
            one.

        Raises:
            ValueError: If ``name`` is not an embedder component.

        Example:
            >>> embedder = await composition.local_embedder("embedder")
        """
        component = self._typed(name, EmbedderSettings)
        return await self._factory.open_local(EMBEDDER_OPENERS, name, component)

    async def local_reranker(self, name: str) -> BaseReranker:
        """Open the named reranker in this process, whatever its backend says.

        A model server is what the ``served`` backend calls, so it always loads the
        local model.

        Args:
            name: A reranker component of the workflow file.

        Returns:
            The supplied reranker if one was given, else the cross-encoder one.

        Raises:
            ValueError: If ``name`` is not a reranker component.

        Example:
            >>> reranker = await composition.local_reranker("reranker")
        """
        component = self._typed(name, RerankerSettings)
        return await self._factory.open_local(RERANKER_OPENERS, name, component)

    async def calibration(self, name: str) -> ComponentCalibration:
        """Calibrate one component's in-process model on this machine, on demand.

        The model loads as an App would load it, whatever the backend says, and
        every parameter calibration chooses is searched, pinned or not, under
        the settings' budget and rounds. A ready-made port is not timed.

        Args:
            name: An embedder or reranker component of the workflow file.

        Returns:
            The report, the CPUs and runtime it ran on, and the environment
            paths that pin the choice.

        Raises:
            ValueError: If ``name`` is no component, or one that runs no model
                in-process.

        Example:
            >>> calibration = await composition.calibration("reranker")
            >>> calibration.pinned
            {'QP_COMPONENTS__RERANKER__LOCAL__THREADS': 2, ...}
        """
        component, local_models = self._component(name), self._local_models
        if isinstance(component, EmbedderSettings):
            return await calibrate_embedder_on_local(name, component, local_models)
        if isinstance(component, RerankerSettings):
            return await calibrate_reranker_on_local(name, component, local_models)
        raise ValueError(
            f"component {name!r} is of type {component.type!r}: only an embedder "
            "or a reranker runs a model in this process"
        )

    async def _open(self, name: str) -> object:
        component = self._component(name)
        if isinstance(component, RegisteredComponentSettings):
            resources = ComponentResources(
                name=name,
                type_settings=component.type_block(),
                component_settings=self._component,
            )
            return await self._factory.open_registered(name, component, resources)
        return await self._factory.open(
            OPENERS_BY_TYPE[component.type], name, component
        )

    def _component(self, name: str) -> ComponentSettings:
        component = self._components.get(name)
        if component is None:
            known = ", ".join(map(repr, self._components))
            raise ValueError(f"no component named {name!r} (the components: {known})")
        return component

    def _typed(self, name: str, kind: type[C]) -> C:
        component = self._component(name)
        if not isinstance(component, kind):
            expected = kind.model_fields["type"].default
            raise ValueError(
                f"component {name!r} is of type {component.type!r}, "
                f"not {expected!r}"
            )
        return component

    async def _store_embedder(self) -> BaseEmbedder:
        # What seeds a fake store, so its vectors are the ones queries get.
        return await self._embedder(self.settings.ingestion_embedder())

    async def _embedder(self, name: str) -> BaseEmbedder:
        component = self._typed(name, EmbedderSettings)
        return await self._factory.open(EMBEDDER_OPENERS, name, component)

    async def _vector_store(self) -> BaseVectorStore:
        name = self.settings.ingestion_store()
        store = self._typed(name, VectorStoreSettings)
        return await self._factory.open(VECTOR_STORE_OPENERS, name, store)
