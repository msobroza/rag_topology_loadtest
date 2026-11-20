"""Runtime configuration: the workflow file, ``QP_*`` variables and a ``.env`` file.

The workflow file declares the workflow: its ``components`` and its ``graph``
(``files/workflow_file.py``), and the settings the components share. Each top-level
key of the file is a field here, so each of its values has a variable, its
path's levels joined by ``__``: ``QP_COMPONENTS__RERANKER__LOCAL__THREADS``,
``QP_GRAPH__STAGES__RERANK__TOP_K``, ``QP_HTTP__RETRIES``. A component, the
vector store as well as a model, is named, has a type that defaults to its
name, states its backend and has one block of settings per backend
(``components/component_settings.py``). The shared settings are four blocks
named after what they are about, ``local_models``, ``calibration``, ``http``
and ``domino`` (``shared_settings.py``). An endpoint's API key is never a
setting: its ``key`` block names the key source it comes from.

Each component runs on the backend its settings give, and nothing else decides
it: the deploy tool writes the backend and URL of each component a topology
alias places into the workflow App's launch script (ADR 0003), and on a laptop
a component is moved by setting its backend and its URL. ``QP_TOPOLOGY_ALIAS``
is only a label, the topology alias the deploy tool deployed, echoed as the
``X-RAG-Topology`` header. A component on ``served`` or ``api`` without a URL,
or a run label or topology alias no header can carry, fails construction, so at
start-up. Domino's own run variables are read under the names Domino gives
them.

Example:
    >>> settings = WorkflowSettings()  # from the env and the workflow file
    >>> settings = WorkflowSettings(  # in tests: merged into the file's components
    ...     _env_file=None, components={"reranker": {"backend": "fake"}})

Every tunable is a field here, a setting of a component, a shared setting or a
setting of a stage in the graph, so one ``QP_*`` variable changes it; adapters
receive the values from the ``composition`` package and never read this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ..domain.components.routes import Route
from ..domain.graph_description import GraphDescription
from ..domain.prompts import Prompt
from ..domain.run_label import (
    RUN_LABEL_PATTERN,
    RUN_LABEL_SHAPE,
    default_run_label,
    validate_run_label,
)
from .components.component_settings import (
    Component,
    EmbedderSettings,
    LlmSettings,
    ModelComponent,
    RerankerSettings,
    VectorStoreSettings,
    typed_by_name,
)
from .components.endpoint_urls import component_endpoint_url, missing_url_message
from .components.key_settings import OldApiKeysRefused
from .components.router_settings import check_router_embedders, check_router_routes
from .extensions import load_extensions
from .files.prompts_file import prompts_file, prompts_file_source
from .files.routes_file import routes_file, routes_file_name, routes_file_source
from .files.workflow_file import workflow_file, workflow_file_source
from .searched_embedder import searched_embedder
from .shared_settings import (
    CalibrationSettings,
    DefaultHttpSettings,
    DominoSettings,
    LocalModelsSettings,
)


class WorkflowSettings(BaseSettings):
    """The whole runtime configuration, one field per top-level ``QP_*`` variable.

    Construction fills the default run label and checks the endpoints and the
    labels, so a bad configuration fails at start-up. The settings of a stage
    are checked by its stage type, when the graph is built at start-up.

    Example:
        >>> WorkflowSettings(components={"reranker": {"backend": "served",
        ...     "served": {"url": "http://host:8001/v3/rerank"}}}
        ... ).run_label
        'embedder=local+reranker=served+llm=api+vector_store=elasticsearch'
    """

    model_config = SettingsConfigDict(
        env_prefix="QP_",
        env_file=".env",
        # Between the levels of a value of the workflow file, in its variable.
        env_nested_delimiter="__",
        # Other QP_* variables are read by other settings (QP_LOADTEST_*,
        # QP_DEPLOY_*); the workflow file refuses an unknown key itself.
        extra="ignore",
        protected_namespaces=(),
    )

    # --- the workflow ----------------------------------------------------------------
    # The packages of the use case whose every module is imported before the
    # components are read, so that the classes they define register (ADR 0006).
    extensions: list[str] = Field(default_factory=list)
    # The workflow file's graph block: its stages with their own settings (the
    # retrieval mode, the sizes, the prompts), its edges and conditional edges.
    graph: GraphDescription
    # The library that runs the graph along its edges, a name registered in
    # ORCHESTRATORS (ADR 0010); the graph does not change with it.
    orchestrator: str = "langgraph"
    # The workflow file's components block: each model and the vector store by
    # name, its backend, its settings and one block per backend.
    components: dict[str, Component]
    # The prompts file's prompts by name, which a stage names (files/prompts_file.py).
    prompts: dict[str, Prompt] = Field(default_factory=dict)
    # The routes file's catalogue, of which each router lists some
    # (files/routes_file.py).
    routes: list[Route] = Field(default_factory=list)

    # --- the topology alias and the run label: labels, they decide no backend -------
    # The topology alias the deploy tool deployed, sent as X-RAG-Topology; unset,
    # no topology is reported.
    topology_alias: str | None = None
    # "" = the default run label (each component with its backend)
    run_label: str = ""

    # --- what the components share: top-level blocks of the workflow file ------------
    local_models: LocalModelsSettings = Field(default_factory=LocalModelsSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    http: DefaultHttpSettings = Field(default_factory=DefaultHttpSettings)
    domino: DominoSettings = Field(default_factory=DominoSettings)

    # --- Domino's own run variables --------------------------------------------------
    # Not settings of ours: read here and nowhere else. Aliases never get
    # the QP_ prefix; the lower-case choice keeps the field name usable as a kwarg.
    domino_api_proxy: str = Field(
        default="",
        validation_alias=AliasChoices("DOMINO_API_PROXY", "domino_api_proxy"),
    )
    domino_hardware_tier_id: str = Field(
        default="",
        validation_alias=AliasChoices(
            "DOMINO_HARDWARE_TIER_ID", "domino_hardware_tier_id"
        ),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order the sources: init kwargs > environment > .env > the three files.

        The class defaults come last. The sources are merged key by key, so a
        variable such as ``QP_GRAPH__STAGES__RERANK__TOP_K`` overrides one value
        of the file and keeps the others.

        Args:
            settings_cls: The settings class being built.
            init_settings: Constructor arguments.
            env_settings: ``QP_*`` environment variables.
            dotenv_settings: The ``.env`` file.
            file_secret_settings: Not read: no secrets directory is configured.

        Returns:
            The sources, highest priority first.
        """
        # Pydantic passes every source by keyword, so the names stay.
        del file_secret_settings
        # Read after the init kwargs, the environment and .env, so it sees them.
        workflow = OldApiKeysRefused(
            workflow_file_source(settings_cls, workflow_file()), cls.variable_of
        )
        prompts = prompts_file_source(settings_cls, prompts_file())
        routes = routes_file_source(settings_cls, routes_file())
        return init_settings, env_settings, dotenv_settings, workflow, prompts, routes

    @model_validator(mode="before")
    @classmethod
    def load_the_extensions(cls, data: Any) -> Any:
        """Import the packages ``extensions`` names, before the components are read.

        Args:
            data: Every source merged, as read.

        Returns:
            ``data``, unchanged.
        """
        if isinstance(data, Mapping):
            load_extensions(data.get("extensions"))
        return data

    @field_validator("components", mode="before")
    @classmethod
    def type_defaults_to_name(cls, components: object) -> object:
        """Give each component without a ``type`` its own name as its type.

        Args:
            components: The ``components`` block, merged from every source.

        Returns:
            The block, each entry typed.
        """
        return typed_by_name(components)

    @field_validator("topology_alias")
    @classmethod
    def header_safe_topology_alias(cls, alias: str | None) -> str | None:
        """Check that the topology alias can be sent as a header; empty is unset.

        Args:
            alias: The topology alias, None when unset.

        Returns:
            ``alias``, or None when it is unset or empty.

        Raises:
            ValueError: If ``alias`` is not header-safe; the message names its
                variable, the value and the shape expected.
        """
        if not alias:
            return None
        if RUN_LABEL_PATTERN.fullmatch(alias):
            return alias
        variable = cls.variable_of("topology_alias")
        raise ValueError(
            f"{variable}={alias!r} cannot be sent as the X-RAG-Topology header: "
            f"a topology alias is {RUN_LABEL_SHAPE}"
        )

    @model_validator(mode="after")
    def fill_run_label(self) -> WorkflowSettings:
        """Check the run label given, or fill the default one.

        Returns:
            The settings, ``run_label`` the one given, else each component with
            its backend.
        """
        if self.run_label:
            self.run_label = validate_run_label(self.run_label)
            return self
        backends = {name: c.backend for name, c in self.components.items()}
        self.run_label = default_run_label(backends)
        return self

    @model_validator(mode="after")
    def check_endpoint_urls(self) -> WorkflowSettings:
        """Fail at start-up when a component on ``served`` or ``api`` has no URL.

        The URL is the one of the block named after the backend, a registered
        type's on ``served`` included; the URL of a component on another
        backend is not checked.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If the URL of a ``served`` or ``api`` component does not
                start with ``http://`` or ``https://``; the message names the
                variables, the value and an example.
        """
        for name, component in self.components.items():
            url = component_endpoint_url(component)
            if url is not None and not url.startswith(("http://", "https://")):
                backend = self.variable_of("components", name, "backend")
                url_variable = self.variable_of(
                    "components", name, component.backend, "url"
                )
                raise ValueError(
                    missing_url_message(component, backend, url_variable, url)
                )
        return self

    @model_validator(mode="after")
    def check_routers(self) -> WorkflowSettings:
        """Fail at start-up when a router names no embedder, or a route the file lacks.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If a router's ``embedder`` names no embedder component
                (the message names the value and the embedders declared), a
                router lists a route the catalogue lacks, or two routes share a
                name (the message names the router, the route, the routes file
                and its routes).
        """
        check_router_embedders(self.components)
        where = routes_file_name(routes_file())
        check_router_routes(self.components, self.routes, where)
        return self

    @model_validator(mode="after")
    def check_store_names(self) -> WorkflowSettings:
        """Fail at start-up when a store names no index or collection to keep.

        Where the passages are kept is the use case's choice, so the toolkit
        has no default for it: the block of the store's backend states it.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If a vector store on ``elasticsearch`` has no index, or
                on ``chroma`` no collection; the message names the variables,
                the workflow file's key and the value.
        """
        for name, component in self.vector_stores().items():
            named = component.kept_in()
            if named is None or named[1]:
                continue
            key, backend = named[0], component.backend
            backend_variable = self.variable_of("components", name, "backend")
            variable = self.variable_of("components", name, backend, key)
            raise ValueError(
                f"{backend_variable}={backend!r} needs {variable}, the {key} the "
                f"passages are kept in (components.{name}.{backend}.{key} in the "
                "workflow file); got ''"
            )
        return self

    @model_validator(mode="after")
    def check_served_aliases(self) -> WorkflowSettings:
        """Fail at start-up when a component on ``served`` has no alias.

        A model server of ours serves a model under its alias, the model name a
        served component sends. An LLM's, like any model's, defaults to the
        last part of its source id, which the toolkit's defaults leave empty.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If a ``served`` component's alias is empty; the message
                names the variables to set.
        """
        for name, component in self.model_components().items():
            if component.backend == "served" and not component.alias:
                backend = self.variable_of("components", name, "backend")
                alias = self.variable_of("components", name, "alias")
                source_id = self.variable_of("components", name, "source_id")
                raise ValueError(
                    f"{backend}='served' needs {alias}, the name its model server "
                    f"serves it under (or {source_id}, whose last part it "
                    "defaults to); got ''"
                )
        return self

    @classmethod
    def variable_of(cls, field: str, *levels: str) -> str:
        """Return the environment variable a value is read from: its path.

        Args:
            field: The field name, the path's first level.
            *levels: The levels below it, for a value of the workflow file.

        Returns:
            Its alias when it has one (Domino's own variables), else the prefix
            and the path's levels in capitals, joined by ``__``.

        Raises:
            ValueError: If ``field`` is not a field of these settings.

        Example:
            >>> WorkflowSettings.variable_of(
            ...     "components", "reranker", "local", "threads")
            'QP_COMPONENTS__RERANKER__LOCAL__THREADS'
        """
        if field not in cls.model_fields:
            raise ValueError(
                f"WorkflowSettings has no field {field!r}; expected one of its fields, "
                "such as 'components'"
            )
        alias = cls.model_fields[field].validation_alias
        if isinstance(alias, AliasChoices):
            return str(alias.choices[0])
        prefix = str(cls.model_config.get("env_prefix", ""))
        return prefix + "__".join((field, *levels)).upper()

    def model_components(self) -> dict[str, ModelComponent]:
        """Return the components that are models, by name: every one but a store.

        Returns:
            The embedders, rerankers and LLMs of the workflow file, in its order.

        Example:
            >>> list(WorkflowSettings().model_components())
            ['embedder', 'reranker', 'llm']
        """
        models = (EmbedderSettings, RerankerSettings, LlmSettings)
        return {n: c for n, c in self.components.items() if isinstance(c, models)}

    def vector_stores(self) -> dict[str, VectorStoreSettings]:
        """Return the components that are vector stores, by name.

        Returns:
            The vector stores of the workflow file, in its order.

        Example:
            >>> list(WorkflowSettings().vector_stores())
            ['vector_store']
        """
        found = self.components.items()
        return {n: c for n, c in found if isinstance(c, VectorStoreSettings)}

    def component_types(self) -> dict[str, str]:
        """Return the type of every component a stage can call, by name.

        Returns:
            Each component's name and type, the vector store's included.

        Example:
            >>> WorkflowSettings().component_types()["vector_store"]
            'vector_store'
        """
        return {name: c.type for name, c in self.components.items()}

    def ingestion_store(self) -> str:
        """Return the vector store that ingestion fills: the only one declared.

        Returns:
            The component's name.

        Raises:
            ValueError: If the workflow file declares no vector store or several.

        Example:
            >>> WorkflowSettings().ingestion_store()
            'vector_store'
        """
        stores = list(self.vector_stores())
        if len(stores) != 1:
            known = ", ".join(map(repr, stores)) or "none"
            raise ValueError(
                "ingestion fills one vector store component, and the workflow "
                f"file declares {len(stores)} (its vector stores: {known})"
            )
        return stores[0]

    def ingestion_embedder(self, name: str | None = None) -> str:
        """Return the embedder that fills the store: the one named, else the only one.

        Ingestion embeds passages with one embedder of the workflow file;
        with several, it must be named.

        Args:
            name: The embedder asked for; None for the only one declared.

        Returns:
            The component's name.

        Raises:
            ValueError: If ``name`` is not an embedder component, or none is
                given and the file declares no embedder or several.

        Example:
            >>> WorkflowSettings().ingestion_embedder()
            'embedder'
        """
        embedders = [n for n, c in self.components.items() if c.type == "embedder"]
        known = ", ".join(map(repr, embedders)) or "none"
        if name is not None and name not in embedders:
            raise ValueError(
                f"{name!r} is not an embedder component of the workflow file "
                f"(its embedders: {known})"
            )
        if name is None and len(embedders) != 1:
            raise ValueError(
                "ingestion embeds with one embedder component, and the workflow "
                f"file declares {len(embedders)} (its embedders: {known}): name "
                "one, with qp-ingest --embedder"
            )
        return name or embedders[0]

    def searched_embedder(self) -> str | None:
        """Return the embedder whose vectors the store is searched with, if one.

        Returns:
            The embedder called by the stage that writes what a ``retrieve``
            stage reads; None when no stage does, or several embedders do.

        Example:
            >>> WorkflowSettings().searched_embedder()
            'embedder'
        """
        return searched_embedder(self.graph, self.component_types())
