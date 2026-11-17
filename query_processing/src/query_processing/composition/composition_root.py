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
  its implementations on their class line, an extension's, by its own class;
- ``ovms_calibration.py``: an OVMS App's stream and thread counts, and the
  folder it serves from.

The openers take each component's settings in the workflow file's shape
(``config/component_settings.py``).

Open ports from one task (an app lifespan or a CLI ``main``): first requests are
not guarded against running concurrently.

Example:
    >>> async with CompositionRoot(WorkflowSettings()) as composition:
    ...     deps = await composition.workflow_dependencies()
    ...     graph = composition.settings.graph  # from the workflow file
    ...     context = stage_context(composition.settings)  # the prompts
    ...     workflow = build_workflow(graph, deps.components(), context)
    ...     result = await workflow.run("How many vacation days?")
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple, TypeVar

import httpx
from pydantic import BaseModel

from ..adapters.embedder.base_embedder import BaseEmbedder
from ..adapters.reranker.base_reranker import BaseReranker
from ..adapters.vector_store.base_vector_store import BaseVectorStore
from ..application.graph_check import check_graph
from ..application.rag_dependencies import RagDependencies
from ..application.stages.base_stage import StageContext
from ..config.component_settings import (
    ComponentSettings,
    EmbedderSettings,
    RerankerSettings,
    VectorStoreSettings,
)
from ..config.ovms_settings import OvmsSettings
from ..config.registered_component_settings import RegisteredComponentSettings
from ..config.router_settings import RouterSettings, routes_of
from ..config.searched_embedder import seeding_embedder
from ..config.workflow_settings import WorkflowSettings
from ..domain.base_component import BaseComponent
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
from .ovms_calibration import (
    OvmsAppCalibration,
    OvmsCalibration,
    OvmsServedComponent,
    require_asked_weight_format,
)
from .registered_components import local_settings
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
        self._ovms = OvmsCalibration(http, local_models, settings.calibration)

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

    async def local_component(self, name: str) -> BaseComponent[Any, Any]:
        """Open the named registered component in this process, for its model server.

        It opens on the backend the workflow file gives it, unless that is
        ``served``, the backend that calls the model server itself: then on
        ``local``.

        Args:
            name: A component of a type registered in ``COMPONENT_TYPES``.

        Returns:
            The supplied component if one was given, else its implementation.

        Raises:
            ValueError: If ``name`` is no such component.

        Example:
            >>> shouter = await composition.local_component("shouter")
        """
        component = self._component(name)
        if not isinstance(component, RegisteredComponentSettings):
            raise ValueError(
                f"component {name!r} is of type {component.type!r}, which has no "
                "call route (expected a component of a type registered on its "
                "class line)"
            )
        local = local_settings(name, component)
        return await self._factory.open_registered(
            name, local, self._resources(name, local)
        )

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

    async def ovms_serving_config(
        self, name: str, ovms: OvmsSettings, models_dir: Path, serving_dir: Path
    ) -> Path:
        """Write the folder an OVMS App serves one component from, calibrating first.

        Each of ``QP_OVMS_NUM_STREAMS`` and ``QP_OVMS_THREADS`` left at 0 is
        calibrated, one ``ovms`` start per candidate (``ovms`` on ``PATH``);
        with both pinned nothing is started. An export of another weight
        format than the settings ask is refused first.

        Args:
            name: An embedder or reranker component of the workflow file.
            ovms: The OVMS App's own settings.
            models_dir: The folder OVMS's exporter wrote into; never written.
            serving_dir: The App's own folder, written for the choice.

        Returns:
            The config file to start OVMS with.

        Raises:
            ValueError: If ``name`` is no component, or no embedder or reranker,
                or the export holds another weight format than the settings'
                ``local_models.openvino_weight_format``.

        Example:
            >>> await composition.ovms_serving_config(
            ...     "reranker", OvmsSettings(), models, serving)
            PosixPath('/tmp/serving/config_reranker.json')
        """
        served = self._served_by_ovms(name, models_dir)
        asked = self.settings.local_models.openvino_weight_format
        require_asked_weight_format(served, asked)
        return await self._ovms.serving_config(served, ovms, serving_dir)

    async def ovms_calibration(
        self, name: str, ovms: OvmsSettings, models_dir: Path
    ) -> OvmsAppCalibration:
        """Calibrate OVMS on one component here, on demand, both parameters searched.

        Args:
            name: An embedder or reranker component of the workflow file.
            ovms: The OVMS App's own settings, for the start timeout.
            models_dir: The folder OVMS's exporter wrote into; never written.

        Returns:
            The report, the CPUs, ``ovms`` as the runtime, and the App's
            variables that pin the choice.

        Raises:
            ValueError: If ``name`` is no component, or no embedder or reranker.

        Example:
            >>> (await composition.ovms_calibration("reranker", ovms, models)).pinned
            {'QP_OVMS_NUM_STREAMS': 2, 'QP_OVMS_THREADS': 2}
        """
        served = self._served_by_ovms(name, models_dir)
        return await self._ovms.calibration(served, ovms)

    def _served_by_ovms(self, name: str, models_dir: Path) -> OvmsServedComponent:
        component = self._component(name)
        if isinstance(component, (EmbedderSettings, RerankerSettings)):
            return OvmsServedComponent(name, component, models_dir)
        raise ValueError(
            f"component {name!r} is of type {component.type!r}: OVMS serves an "
            "export of an embedder or a reranker only (it pulls an LLM itself)"
        )

    async def _open(self, name: str) -> object:
        component = self._component(name)
        if isinstance(component, RegisteredComponentSettings):
            resources = self._resources(name, component)
            return await self._factory.open_registered(name, component, resources)
        return await self._factory.open(
            OPENERS_BY_TYPE[component.type], name, component
        )

    def _resources(
        self, name: str, component: RegisteredComponentSettings
    ) -> ComponentResources:
        catalogue = {route.name: route for route in self.settings.routes}
        routes = (
            routes_of(component, catalogue)
            if isinstance(component, RouterSettings)
            else {}
        )
        return ComponentResources(
            name=name,
            type_settings=component.type_block(),
            component_settings=self._component,
            routes=routes,
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
        # What seeds a fake store, so its vectors are the ones queries get
        # (seeding_embedder), else the only one declared (spec 2.5).
        graph, types = self.settings.graph, self.settings.component_types()
        name = seeding_embedder(graph, types) or self.settings.ingestion_embedder()
        return await self._embedder(name)

    async def _embedder(self, name: str) -> BaseEmbedder:
        component = self._typed(name, EmbedderSettings)
        return await self._factory.open(EMBEDDER_OPENERS, name, component)

    async def _vector_store(self) -> BaseVectorStore:
        name = self.settings.ingestion_store()
        store = self._typed(name, VectorStoreSettings)
        return await self._factory.open(VECTOR_STORE_OPENERS, name, store)


def stage_context(settings: WorkflowSettings) -> StageContext:
    """Return what every stage is given, built from the settings.

    Args:
        settings: The workflow settings, their prompts read from the prompts file.

    Returns:
        The stage context: the prompts, by name, and the metadata fields each
        vector store declares, by its component name.

    Example:
        >>> check_graph(settings.graph, settings.component_types(),
        ...             stage_context(settings))
    """
    declared = {
        name: store.metadata_field_info
        for name, store in settings.vector_stores().items()
    }
    return StageContext(prompts=settings.prompts, metadata_field_info=declared)


def check_workflow_graph(settings: WorkflowSettings) -> None:
    """Refuse a workflow graph that cannot run, before any component opens.

    Args:
        settings: The workflow settings, their graph, components and prompts.

    Raises:
        WorkflowGraphError: The graph cannot run (``check_graph``); the message
            names what is wrong and what was expected.

    Example:
        >>> check_workflow_graph(WorkflowSettings())  # a use case's: nothing raised
    """
    check_graph(
        settings.graph,
        settings.component_types(),
        stage_context(settings),
        component_type_settings(settings),
    )


def component_type_settings(settings: WorkflowSettings) -> dict[str, BaseModel | None]:
    """Return the keys of each component's type, by the component's name.

    The start-up checks give a plain call's written fields the labels its
    component type declares from them, such as a router's routes.

    Args:
        settings: The workflow settings.

    Returns:
        Each component's type keys, read with its type's settings class; None
        for a component whose type has none, or that is not registered.

    Example:
        >>> component_type_settings(settings)["router"].routes
        ['self_contained', 'leans_on_history']
    """
    return {
        name: c.type_block() if isinstance(c, RegisteredComponentSettings) else None
        for name, c in settings.components.items()
    }
