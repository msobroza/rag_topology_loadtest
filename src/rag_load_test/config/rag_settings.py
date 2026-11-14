"""Runtime configuration from ``RAG_*`` environment variables (or a ``.env`` file).

Three model components, one block of settings each: embedding, rerank and
generation. A component's backend says how it runs: ``local`` (a model loaded
in this process), ``api`` (an endpoint called over HTTP: a URL, an API format,
a key and a model name), ``fake``, and for rerank also ``none``.
``RAG_TOPOLOGY`` decides the embedding and rerank backends (the rule lives in
``domain/topology.py``); ``RAG_EMBEDDING_BACKEND`` / ``RAG_RERANK_BACKEND`` may
only repeat that choice or pick a variant that loads no model of ours. A
contradiction, an ``api`` backend without a URL or a bad ``RAG_RUN_LABEL`` fails
construction, so at start-up.

Example:
    >>> settings = RagSettings()  # from the env
    >>> settings = RagSettings(_env_file=None, rerank_backend="fake")  # in tests

Every tunable is a field here, so one ``RAG_*`` variable changes it; adapters
receive the values from ``adapters/rag_ports.py`` and never read this module. The
prompts come from a YAML file, the lowest-priority source (see
``prompts_source.py``).
"""

from __future__ import annotations

from typing import cast

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ..domain.api_endpoint import (
    ApiEndpoint,
    EmbeddingApiFormat,
    GenerationApiFormat,
    RerankApiFormat,
)
from ..domain.backends import (
    DEFAULT_GENERATION_BACKEND,
    DEFAULT_VECTOR_STORE_BACKEND,
    LEXICAL_STORE_BACKENDS,
    EmbeddingBackend,
    GenerationBackend,
    LocalModelRuntime,
    RerankBackend,
    VectorStoreBackend,
)
from ..domain.prompts import RagPrompts
from ..domain.run_label import default_run_label, validate_run_label
from ..domain.topology import (
    Topology,
    resolve_embedding_backend,
    resolve_rerank_backend,
)
from ..domain.workflow_parameters import (
    DEFAULT_RETRIEVAL_MODE,
    RetrievalMode,
    WorkflowParameters,
)
from .prompts_source import prompts_path, prompts_source

# Shown in the error for an ``api`` backend without a URL: one route per component.
_API_URL_EXAMPLES = {
    "embedding": "http://localhost:8002/v3/embeddings",
    "rerank": "http://localhost:8001/v3/rerank",
    "generation": "https://api.openai.com/v1/chat/completions",
}


class RagSettings(BaseSettings):
    """The whole runtime configuration, one field per ``RAG_*`` variable.

    Construction resolves the backends from the topology, fills the default run
    label and checks the endpoints, retrieval sizes and prompts, so a bad
    configuration fails at start-up.

    Example:
        >>> RagSettings(topology="split-reranker",
        ...             rerank_api_url="http://host:8001/v3/rerank").rerank_backend
        'api'
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- topology and run label ----------------------------------------------------
    topology: Topology = "monolith"
    # "" = the default run label (topology plus every choice off its default)
    run_label: str = ""

    # --- embedding -------------------------------------------------------------------
    # None until resolve_topology fills it from the topology; never None after, so
    # the annotation stays non-optional for rag_ports.py. validate_default=False:
    # pydantic-settings would reject the placeholder; the cast satisfies pydantic's
    # typing, which wants an unvalidated default to match the annotation.
    embedding_backend: EmbeddingBackend = Field(
        default=cast(EmbeddingBackend, None), validate_default=False
    )
    # The local model (a Hugging Face id or a folder); our model servers load it too.
    embedding_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    # Put in front of every question before it is embedded (arctic-embed expects
    # it on queries, never on passages); "" for a model that takes none. The
    # fake backend gets none: see workflow_parameters.
    embedding_query_prefix: str = "query: "
    embedding_api_url: str = ""  # the full route, e.g. http://host:8002/v3/embeddings
    embedding_api_format: EmbeddingApiFormat = "openai"
    # "" = no Authorization header, "domino" = the run's token, else a bearer token
    embedding_api_key: str = ""
    # The name sent; our model servers serve embedding_model under it.
    embedding_api_model: str = "snowflake-arctic-embed-l-v2.0"
    embedding_batch_size: int = Field(default=64, ge=1)  # texts per call
    embedding_threads: int = Field(default=0, ge=0)  # 0 = from the thread budget

    # --- rerank ----------------------------------------------------------------------
    rerank_backend: RerankBackend = Field(
        default=cast(RerankBackend, None), validate_default=False
    )
    rerank_model: str = "antoinelouis/crossencoder-camembert-base-mmarcoFR"
    rerank_api_url: str = ""  # the full route, e.g. http://host:8001/v3/rerank
    rerank_api_format: RerankApiFormat = "cohere"
    rerank_api_key: str = ""  # as embedding_api_key
    rerank_api_model: str = "crossencoder-camembert-base-mmarcoFR"
    rerank_batch_size: int = Field(default=32, ge=1)  # passages per local call
    rerank_threads: int = Field(default=0, ge=0)  # 0 = from the thread budget

    # --- generation ------------------------------------------------------------------
    generation_backend: GenerationBackend = DEFAULT_GENERATION_BACKEND
    generation_api_url: str = "https://api.openai.com/v1/chat/completions"
    generation_api_format: GenerationApiFormat = "openai"
    generation_api_key: str = ""  # as embedding_api_key
    generation_api_model: str = "gpt-4o-mini"
    generation_temperature: float = Field(default=0.0, ge=0, le=2)
    # An answer takes longer than a model-server call, so generation has its own
    # timeout; retries and backoff are the shared http_retries / http_backoff_s.
    generation_timeout_s: float = Field(default=120.0, gt=0)

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

    # --- retrieval and prompts (no prompt default: see prompts.yaml) -----------------
    retrieval_mode: RetrievalMode = DEFAULT_RETRIEVAL_MODE  # dense | lexical | hybrid
    hybrid_rrf_k: int = Field(default=60, ge=1)  # rank constant of the hybrid fusion
    top_k_retrieve: int = Field(default=20, ge=1)  # candidates from the store
    top_k_rerank: int = Field(default=5, ge=1)  # passages kept and sent to the LLM
    system_prompt: str
    user_prompt: str  # holds {context} and {question}

    # --- in-process models and threads -----------------------------------------------
    local_model_runtime: LocalModelRuntime = "openvino"  # openvino | torch
    # Where the openvino runtime keeps its models, one folder each. A model
    # without a folder there is downloaded and converted into it the first time
    # it loads; rag-export-openvino fills it ahead of time.
    openvino_models_dir: str = "models/openvino"
    model_workers: int = Field(default=4, ge=1)  # inference calls running at once, max
    # Virtual CPUs split between workers (domain/thread_budget.py): 0 = find
    # out; > 0 wins over every other count.
    model_cpus: float = Field(default=0, ge=0)

    # --- HTTP and Domino -------------------------------------------------------------
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
        """Order the sources: init kwargs > environment > .env > secrets > prompts YAML.

        Args:
            settings_cls: The settings class being built.
            init_settings: Constructor arguments.
            env_settings: ``RAG_*`` environment variables.
            dotenv_settings: The ``.env`` file.
            file_secret_settings: The secrets directory.

        Returns:
            The sources, highest priority first.
        """
        prompts = prompts_source(settings_cls, prompts_path())
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            prompts,
        )

    @model_validator(mode="after")
    def resolve_topology(self) -> RagSettings:
        """Fill the embedding and rerank backends from the topology, and the run label.

        Returns:
            The settings, with ``embedding_backend``, ``rerank_backend`` and
            ``run_label`` resolved.
        """
        self.embedding_backend = resolve_embedding_backend(
            self.topology, self.embedding_backend
        )
        self.rerank_backend = resolve_rerank_backend(self.topology, self.rerank_backend)
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
                ``http://`` or ``https://``; the message names the variable, the
                value and an example.
        """
        components = (
            ("embedding", self.embedding_backend, self.embedding_api_url),
            ("rerank", self.rerank_backend, self.rerank_api_url),
            ("generation", self.generation_backend, self.generation_api_url),
        )
        for component, backend, url in components:
            if backend == "api" and not url.startswith(("http://", "https://")):
                raise ValueError(_missing_url_message(component, url))
        return self

    @model_validator(mode="after")
    def check_workflow_parameters(self) -> RagSettings:
        """Fail at start-up, not on the first query, on bad sizes or prompts.

        ``RagPrompts`` checks the user template and raises its own ``ValueError``.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If ``top_k_rerank`` exceeds ``top_k_retrieve``: the
                reranker can keep at most the candidates retrieval returns.
        """
        if self.top_k_rerank > self.top_k_retrieve:
            raise ValueError(
                f"RAG_TOP_K_RERANK={self.top_k_rerank} exceeds "
                f"RAG_TOP_K_RETRIEVE={self.top_k_retrieve}: the reranker can keep "
                "at most the candidates retrieval returns"
            )
        self.workflow_parameters()  # RagPrompts checks the user template
        return self

    @model_validator(mode="after")
    def check_retrieval_store(self) -> RagSettings:
        """Fail at start-up when lexical or hybrid retrieval has no store ranking text.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If ``retrieval_mode`` is ``lexical`` or ``hybrid`` and
                ``vector_store_backend`` is not in ``LEXICAL_STORE_BACKENDS``.
        """
        if self.retrieval_mode == "dense":
            return self
        if self.vector_store_backend in LEXICAL_STORE_BACKENDS:
            return self
        raise ValueError(
            f"RAG_RETRIEVAL_MODE={self.retrieval_mode!r} needs a store that ranks "
            f"text: RAG_VECTOR_STORE_BACKEND={self.vector_store_backend!r} has none "
            "(Chroma ranks text on Chroma Cloud only); use "
            f"{' or '.join(map(repr, LEXICAL_STORE_BACKENDS))}, or "
            "RAG_RETRIEVAL_MODE=dense"
        )

    def workflow_parameters(self) -> WorkflowParameters:
        """Build what ``RagWorkflow`` runs with: retrieval mode, sizes and prompts.

        The query prefix belongs to the embedding model, so the ``fake`` embedder,
        which runs none, gets "".

        Returns:
            The workflow parameters from the retrieval settings and the two
            prompts.

        Example:
            >>> RagWorkflow(deps, RagSettings().workflow_parameters())
        """
        is_fake = self.embedding_backend == "fake"
        return WorkflowParameters(
            top_k_retrieve=self.top_k_retrieve,
            top_k_rerank=self.top_k_rerank,
            prompts=RagPrompts(system=self.system_prompt, user=self.user_prompt),
            retrieval_mode=self.retrieval_mode,
            rrf_k=self.hybrid_rrf_k,
            query_prefix="" if is_fake else self.embedding_query_prefix,
        )

    def embedding_api(self) -> ApiEndpoint:
        """Return the embedding endpoint, from the four ``embedding_api_*`` fields.

        Returns:
            The endpoint's URL, API format, key and model name.

        Example:
            >>> RagSettings(embedding_api_url="http://h/v3/embeddings").embedding_api()
            ApiEndpoint(url='http://h/v3/embeddings', api_format='openai', ...)
        """
        return ApiEndpoint(
            url=self.embedding_api_url,
            api_format=self.embedding_api_format,
            api_key=self.embedding_api_key,
            model=self.embedding_api_model,
        )

    def rerank_api(self) -> ApiEndpoint:
        """Return the rerank endpoint, from the four ``rerank_api_*`` fields.

        Returns:
            The endpoint's URL, API format, key and model name.

        Example:
            >>> RagSettings(rerank_api_url="http://h/v3/rerank").rerank_api().api_format
            'cohere'
        """
        return ApiEndpoint(
            url=self.rerank_api_url,
            api_format=self.rerank_api_format,
            api_key=self.rerank_api_key,
            model=self.rerank_api_model,
        )

    def generation_api(self) -> ApiEndpoint:
        """Return the generation endpoint, from the four ``generation_api_*`` fields.

        Returns:
            The endpoint's URL, API format, key and model name.

        Example:
            >>> RagSettings().generation_api().url
            'https://api.openai.com/v1/chat/completions'
        """
        return ApiEndpoint(
            url=self.generation_api_url,
            api_format=self.generation_api_format,
            api_key=self.generation_api_key,
            model=self.generation_api_model,
        )

    def _default_label(self) -> str:
        return default_run_label(
            self.topology,
            embedding=self.embedding_backend,
            rerank=self.rerank_backend,
            store=self.vector_store_backend,
            generation=self.generation_backend,
            retrieval=self.retrieval_mode,
        )


def _missing_url_message(component: str, url: str) -> str:
    # One message shape for the three components: variable, value, example.
    prefix = f"RAG_{component.upper()}"
    return (
        f"{prefix}_BACKEND='api' needs {prefix}_API_URL, the full route of the "
        f"endpoint (for example {_API_URL_EXAMPLES[component]}); got {url!r}"
    )
