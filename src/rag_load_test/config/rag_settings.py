"""Runtime configuration: the workflow file, ``RAG_*`` variables and a ``.env`` file.

The workflow file declares the workflow: its ``components`` and its ``graph``
(``workflow_file.py``). Each top-level key of the file is a field here, so each
of its values has a variable, its path's levels joined by ``__``:
``RAG_COMPONENTS__RERANKER__LOCAL__THREADS``,
``RAG_GRAPH__STAGES__RERANK__TOP_K``. A component is named, has a type that
defaults to its name, states its backend and has one block of settings per
backend (``component_settings.py``). An API key is read from the environment,
by its path, such as ``RAG_COMPONENTS__LLM__API__API_KEY``.

``RAG_TOPOLOGY`` places the components named ``embedder`` and ``reranker``:
one on ``local`` is moved behind a model server (``api``) when the topology
says so (the rule lives in ``domain/topology.py``). A component on ``api``
without a URL or a bad ``RAG_RUN_LABEL`` fails construction, so at start-up.
The vector store and the settings the components share are still flat
variables.

Example:
    >>> settings = RagSettings()  # from the env and the workflow file
    >>> settings = RagSettings(  # in tests: merged into the file's components
    ...     _env_file=None, components={"reranker": {"backend": "fake"}})

Every tunable is a field here, a setting of a component or a setting of a
stage in the graph, so one ``RAG_*`` variable changes it; adapters receive the
values from the ``composition`` package and never read this module.
"""

from __future__ import annotations

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ..domain.backends import (
    DEFAULT_GENERATION_BACKEND,
    DEFAULT_VECTOR_STORE_BACKEND,
    LocalModelRuntime,
    VectorStoreBackend,
)
from ..domain.graph_description import GraphDescription
from ..domain.retrieval_mode import DEFAULT_RETRIEVAL_MODE
from ..domain.run_label import default_run_label, validate_run_label
from ..domain.topology import (
    TOPOLOGY_BACKENDS,
    PlacedComponent,
    Topology,
    placed_backend,
)
from .component_settings import (
    VECTOR_STORE,
    ChromaSettings,
    ComponentSettings,
    ElasticsearchSettings,
    FakeStoreSettings,
    ModelComponent,
    VectorStoreSettings,
    typed_by_name,
)
from .workflow_file import workflow_file, workflow_file_source

# Shown in the error for an ``api`` backend without a URL: one route per type.
_API_URL_EXAMPLES = {
    "embedder": "http://localhost:8002/v3/embeddings",
    "reranker": "http://localhost:8001/v3/rerank",
    "llm": "https://api.openai.com/v1/chat/completions",
}
# The components a topology places, by name, and the field of TopologyBackends
# that holds their backend; the deploy tool's App parts have the same names.
_PLACED: dict[str, PlacedComponent] = {"embedder": "embedding", "reranker": "rerank"}


class RagSettings(BaseSettings):
    """The whole runtime configuration, one field per top-level ``RAG_*`` variable.

    Construction places the components the topology places, fills the default
    run label and checks the endpoints, so a bad configuration fails at
    start-up. The settings of a stage are checked by its stage type, when the
    graph is built at start-up.

    Example:
        >>> RagSettings(topology="split-reranker", components={"reranker": {
        ...     "api": {"url": "http://host:8001/v3/rerank"}}}
        ... ).components["reranker"].backend
        'api'
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        # Between the levels of a value of the workflow file, in its variable.
        env_nested_delimiter="__",
        # Other RAG_* variables are read by other settings (RAG_LOADTEST_*,
        # RAG_DEPLOY_*); the workflow file refuses an unknown key itself.
        extra="ignore",
        protected_namespaces=(),
    )

    # --- the workflow ----------------------------------------------------------------
    # The workflow file's graph block: its stages with their own settings (the
    # retrieval mode, the sizes, the prompts), its edges and conditional edges.
    graph: GraphDescription
    # The workflow file's components block: each model component by name, its
    # backend, source id and alias, its settings and one block per backend.
    components: dict[str, ModelComponent]

    # --- topology and run label ----------------------------------------------------
    topology: Topology = "monolith"
    # "" = the default run label (topology plus every choice off its default)
    run_label: str = ""

    # --- vector store ----------------------------------------------------------------
    # "fake" = in-memory store seeded with the synthetic corpus when the app starts
    vector_store_backend: VectorStoreBackend = DEFAULT_VECTOR_STORE_BACKEND
    es_url: str = "http://localhost:9200"  # user:password@ in the URL for basic auth
    es_api_key: str = ""
    es_index: str = "rag_passages"
    # Analyzer of the passages' text, applied when the index is created and to
    # every lexical query: "standard", or a language's (e.g. "french"). An
    # index keeps the analyzer it was created with.
    es_text_analyzer: str = "standard"
    chroma_url: str = ""  # a Chroma server; "" = embedded Chroma at chroma_path
    chroma_path: str = "chroma"
    chroma_collection: str = "rag_passages"
    # Same size as ``make ingest`` so a fake-store app answers like an ingested one.
    fake_store_seed_documents: int = Field(default=200, ge=1)

    # --- in-process models and threads -----------------------------------------------
    local_model_runtime: LocalModelRuntime = "openvino"  # openvino | torch
    # Where the openvino runtime keeps its models, one folder each. A model
    # without a folder there is downloaded and converted into it the first time
    # it loads; rag-export-openvino fills it ahead of time.
    openvino_models_dir: str = "models/openvino"
    # Model calls running at once, max, under torch only; openvino runs one call
    # at a time on every CPU (domain/thread_budget.py).
    model_workers: int = Field(default=4, ge=1)
    # Virtual CPUs the model calls share (domain/thread_budget.py): 0 = find
    # out; > 0 wins over every other count.
    model_cpus: float = Field(default=0, ge=0)
    # Calibration of a local model (stage_calibration.py), per stage: no new
    # candidate is started once this many seconds have passed.
    calibration_budget_s: float = Field(default=90.0, gt=0)
    # Timed requests per candidate, after one warm-up.
    calibration_rounds: int = Field(default=1, ge=1)

    # --- HTTP and Domino -------------------------------------------------------------
    # Every call out but the LLM's, whose timeout is its own (components.llm.http).
    http_timeout_s: float = Field(default=30.0, gt=0)
    http_retries: int = Field(default=2, ge=0)  # on transport errors, 5xx and 429
    http_backoff_s: float = Field(default=0.2, ge=0)  # before the 1st retry; doubles
    domino_token_url: str = "http://localhost:8899/access-token"
    domino_token_ttl_s: float = Field(default=240.0, gt=0)  # tokens live ~5 min
    domino_tracing: bool = False
    # Domino's own run variables, read here and nowhere else. Aliases never get
    # the RAG_ prefix; the lower-case choice keeps the field name usable as a kwarg.
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
        """Order the sources: init kwargs > environment > .env > the workflow file.

        The class defaults come last. The sources are merged key by key, so a
        variable such as ``RAG_GRAPH__STAGES__RERANK__TOP_K`` overrides one value
        of the file and keeps the others.

        Args:
            settings_cls: The settings class being built.
            init_settings: Constructor arguments.
            env_settings: ``RAG_*`` environment variables.
            dotenv_settings: The ``.env`` file.
            file_secret_settings: Not read: no secrets directory is configured.

        Returns:
            The sources, highest priority first.
        """
        # Pydantic passes every source by keyword, so the names stay.
        del file_secret_settings
        workflow = workflow_file_source(settings_cls, workflow_file())
        return init_settings, env_settings, dotenv_settings, workflow

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

    @model_validator(mode="after")
    def resolve_topology(self) -> RagSettings:
        """Place the components the topology places, and fill the run label.

        Returns:
            The settings, the ``embedder`` and ``reranker`` components on the
            backend the topology gives them, and ``run_label`` resolved.
        """
        self.components = {
            name: self._placed(name, component)
            for name, component in self.components.items()
        }
        self.run_label = validate_run_label(self.run_label or self._default_label())
        return self

    @model_validator(mode="after")
    def check_api_endpoints(self) -> RagSettings:
        """Fail at start-up when a component on the ``api`` backend has no URL.

        Runs after ``resolve_topology``, so a backend the topology implies counts.
        The URL of a component on another backend is not checked.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If an ``api`` component's URL does not start with
                ``http://`` or ``https://``; the message names the variables,
                the value and an example.
        """
        for name, component in self.components.items():
            url = component.api.url
            if component.backend == "api" and not url.startswith(
                ("http://", "https://")
            ):
                raise ValueError(self._missing_url_message(name, component.type, url))
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
            >>> RagSettings.variable_of("components", "reranker", "local", "threads")
            'RAG_COMPONENTS__RERANKER__LOCAL__THREADS'
        """
        if field not in cls.model_fields:
            raise ValueError(
                f"RagSettings has no field {field!r}; expected one of its fields, "
                "such as 'components'"
            )
        alias = cls.model_fields[field].validation_alias
        if isinstance(alias, AliasChoices):
            return str(alias.choices[0])
        prefix = str(cls.model_config.get("env_prefix", ""))
        return prefix + "__".join((field, *levels)).upper()

    def vector_store_component(self) -> VectorStoreSettings:
        """Return the vector store's settings, in a component's shape.

        Returns:
            Its backend and one block per backend, from the flat variables.

        Example:
            >>> RagSettings().vector_store_component().elasticsearch.index
            'rag_passages'
        """
        return VectorStoreSettings(
            backend=self.vector_store_backend,
            elasticsearch=ElasticsearchSettings(
                url=self.es_url,
                index=self.es_index,
                api_key=self.es_api_key,
                text_analyzer=self.es_text_analyzer,
            ),
            chroma=ChromaSettings(
                url=self.chroma_url,
                path=self.chroma_path,
                collection=self.chroma_collection,
            ),
            fake=FakeStoreSettings(seed_documents=self.fake_store_seed_documents),
        )

    def all_components(self) -> dict[str, ComponentSettings]:
        """Return every component a stage can call, by name: the vector store too.

        Returns:
            The model components of the workflow file, then ``vector_store``.

        Example:
            >>> list(RagSettings().all_components())
            ['embedder', 'reranker', 'llm', 'vector_store']
        """
        return {**self.components, VECTOR_STORE: self.vector_store_component()}

    def component_types(self) -> dict[str, str]:
        """Return the type of every component a stage can call, by name.

        Returns:
            Each component's name and type, the vector store's included.

        Example:
            >>> RagSettings().component_types()["vector_store"]
            'vector_store'
        """
        return {name: c.type for name, c in self.all_components().items()}

    def ingestion_embedder(self, name: str | None = None) -> str:
        """Return the embedder that fills the store: the one named, else the only one.

        Ingestion embeds passages, and the fake store is seeded, with one
        embedder of the workflow file; with several, it must be named.

        Args:
            name: The embedder asked for; None for the only one declared.

        Returns:
            The component's name.

        Raises:
            ValueError: If ``name`` is not an embedder component, or none is
                given and the file declares no embedder or several.

        Example:
            >>> RagSettings().ingestion_embedder()
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
                "one, with rag-ingest --embedder"
            )
        return name or embedders[0]

    def _placed(self, name: str, component: ModelComponent) -> ModelComponent:
        placed = _PLACED.get(name)
        if placed is None:
            return component
        backend = placed_backend(self.topology, placed, component.backend)
        return component.model_copy(update={"backend": backend})

    def _missing_url_message(self, name: str, component_type: str, url: str) -> str:
        # Variables, value, example: the same shape for every component.
        backend = self.variable_of("components", name, "backend")
        url_variable = self.variable_of("components", name, "api", "url")
        return (
            f"{backend}='api' needs {url_variable}, the full route of the "
            f"endpoint (for example {_API_URL_EXAMPLES[component_type]}); "
            f"got {url!r}"
        )

    def _default_label(self) -> str:
        implied = TOPOLOGY_BACKENDS[self.topology]
        backends = {name: c.backend for name, c in self.components.items()}
        return default_run_label(
            self.topology,
            embedding=backends.get("embedder", implied.embedding),
            rerank=backends.get("reranker", implied.rerank),
            store=self.vector_store_backend,
            generation=backends.get("llm", DEFAULT_GENERATION_BACKEND),
            retrieval=self._retrieval_mode(),
        )

    def _retrieval_mode(self) -> str:
        # The label still names the retrieval mode off its default, as it did when
        # the mode was a variable: it is read from the graph's retrieve stage.
        for name, stage in self.graph.stages.items():
            if stage.type_name(name) == "retrieve":
                mode = stage.stage_settings.get("retrieval_mode")
                return str(mode or DEFAULT_RETRIEVAL_MODE)
        return DEFAULT_RETRIEVAL_MODE
