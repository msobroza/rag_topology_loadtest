"""Runtime configuration from ``RAG_*`` environment variables (or a ``.env`` file).

``RAG_TOPOLOGY`` decides the embedder and reranker backends (the rule lives in
``domain/topology.py``); ``RAG_EMBEDDER_BACKEND`` / ``RAG_RERANKER_BACKEND`` may only
repeat that choice or pick a variant that runs no model of ours: a hosted API,
``fake`` (or ``none`` for the reranker). A
contradiction or a bad ``RAG_RUN_LABEL`` fails construction, so at start-up.

Example:
    >>> settings = RagSettings()  # from the env
    >>> settings = RagSettings(_env_file=None, reranker_backend="fake")  # in tests

Every tunable (retrieval mode and sizes, index, batch sizes, timeouts, ...) is a
field here, so one ``RAG_*`` variable changes it; adapters receive the values from
``adapters/rag_ports.py`` and never read this module. The prompts have no Python
default: they come from ``prompts.yaml`` (or the file ``RAG_PROMPTS_FILE`` names),
the lowest-priority source, so ``RAG_SYSTEM_PROMPT`` / ``RAG_USER_PROMPT`` and
constructor arguments still override them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple, cast

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from ..domain.prompts import RagPrompts
from ..domain.topology import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_VECTOR_STORE_BACKEND,
    EmbedderBackend,
    LlmBackend,
    LocalModelRuntime,
    RerankerBackend,
    Topology,
    VectorStoreBackend,
    default_run_label,
    resolve_embedder_backend,
    resolve_reranker_backend,
    validate_run_label,
)
from ..domain.workflow_parameters import (
    DEFAULT_RETRIEVAL_MODE,
    RetrievalMode,
    WorkflowParameters,
)

PACKAGED_PROMPTS = Path(__file__).with_name("prompts.yaml")


class RerankApi(NamedTuple):
    """A hosted rerank endpoint: where it is, the model it is asked for, its key."""

    url: str
    model: str
    api_key: str  # "" sends no Authorization header


class RagSettings(BaseSettings):
    """The whole runtime configuration, one field per ``RAG_*`` variable.

    Construction resolves the backends from the topology, fills the default run
    label and checks the retrieval sizes and prompts, so a bad configuration
    fails at start-up.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    topology: Topology = "monolith"
    # "" = the default run label (topology plus every backend off its default)
    run_label: str = ""

    # None until resolve_topology fills them from the topology; never None after, so
    # the annotation stays non-optional for rag_ports.py. validate_default=False:
    # pydantic-settings would reject the placeholder; the cast satisfies pydantic's
    # typing, which wants an unvalidated default to match the annotation.
    embedder_backend: EmbedderBackend = Field(
        default=cast(EmbedderBackend, None), validate_default=False
    )
    embedder_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    # Put in front of a question before embedder_model embeds it (arctic-embed
    # expects it on queries, never on passages). A hosted or fake embedder has its
    # own conventions and gets none: see workflow_parameters.
    embedder_query_prefix: str = "query: "
    reranker_backend: RerankerBackend = Field(
        default=cast(RerankerBackend, None), validate_default=False
    )
    reranker_model: str = "antoinelouis/crossencoder-camembert-base-mmarcoFR"

    ovms_embeddings_url: str = "http://localhost:8002"
    ovms_embeddings_model: str = "snowflake-arctic-embed-l-v2.0"
    ovms_rerank_url: str = "http://localhost:8001"
    ovms_rerank_model: str = "crossencoder-camembert-base-mmarcoFR"
    # "" = no auth, "domino" = the run's token from localhost:8899, else a bearer token
    ovms_auth: str = ""

    # --- hosted model APIs (embedder_backend=openai, reranker_backend=cohere|openai)
    openai_embedding_model: str = "text-embedding-3-small"
    # Both rerankers speak the Cohere rerank shape; any server with that route works.
    cohere_rerank_url: str = "https://api.cohere.com/v2/rerank"
    cohere_rerank_model: str = "rerank-v3.5"
    # Asked of {OPENAI_BASE_URL}/rerank, a route OpenAI-compatible gateways add
    # (api.openai.com has none), so it has no default.
    openai_rerank_model: str = ""
    # The providers' own variables, read here without the RAG_ prefix ("" leaves
    # the OpenAI SDK to its own environment lookup).
    cohere_api_key: str = Field(
        default="", validation_alias=AliasChoices("COHERE_API_KEY", "cohere_api_key")
    )
    openai_api_key: str = Field(
        default="", validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key")
    )
    openai_base_url: str = Field(
        default="", validation_alias=AliasChoices("OPENAI_BASE_URL", "openai_base_url")
    )

    llm_backend: LlmBackend = DEFAULT_LLM_BACKEND
    openai_model: str = "gpt-4o-mini"  # OPENAI_API_KEY / OPENAI_BASE_URL via the SDK
    llm_temperature: float = Field(default=0.0, ge=0, le=2)

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
    # BM25 of the stores without ranked text search of their own (chroma, fake).
    bm25_k1: float = Field(default=1.2, ge=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    domino_tracing: bool = False

    # --- workflow: retrieval and prompts (no prompt default: see prompts.yaml) -----
    retrieval_mode: RetrievalMode = DEFAULT_RETRIEVAL_MODE  # dense | lexical | hybrid
    hybrid_rrf_k: int = Field(default=60, ge=1)  # rank constant of the hybrid fusion
    top_k_retrieve: int = Field(default=20, ge=1)  # candidates from the store
    top_k_rerank: int = Field(default=5, ge=1)  # passages kept and sent to the LLM
    system_prompt: str
    user_prompt: str  # holds {context} and {question}

    # --- in-process models -------------------------------------------------------
    local_model_runtime: LocalModelRuntime = "openvino"  # openvino | torch
    # Where the openvino runtime keeps its models, one folder each. A model
    # without a folder there is downloaded and converted into it the first time
    # it loads; rag-export-openvino fills it ahead of time.
    openvino_models_dir: str = "models/openvino"
    embedder_batch_size: int = Field(default=64, ge=1)
    reranker_batch_size: int = Field(default=32, ge=1)
    model_workers: int = Field(default=4, ge=1)  # inference calls running at once, max

    # --- thread budget: virtual CPUs split between workers (domain/thread_budget.py)
    model_cpus: float = Field(default=0, ge=0)  # 0 = find out; > 0 wins over all
    embedder_threads: int = Field(default=0, ge=0)  # 0 = from the budget
    reranker_threads: int = Field(default=0, ge=0)  # 0 = from the budget
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

    # --- HTTP to model servers and Domino ------------------------------------------
    http_timeout_s: float = Field(default=30.0, gt=0)
    http_retries: int = Field(default=2, ge=0)  # on transport errors, 5xx and 401
    http_backoff_s: float = Field(default=0.2, ge=0)  # before the 1st retry; doubles
    domino_token_url: str = "http://localhost:8899/access-token"
    domino_token_ttl_s: float = Field(default=240.0, gt=0)  # tokens live ~5 min

    # Same size as ``make ingest`` so a fake-store app answers like an ingested one.
    fake_store_seed_documents: int = Field(default=200, ge=1)

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
        prompts = _prompts_source(settings_cls, _prompts_path())
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            prompts,
        )

    @model_validator(mode="after")
    def resolve_topology(self) -> RagSettings:
        """Fill the model backends from the topology and the default run label.

        Returns:
            The settings, with ``embedder_backend``, ``reranker_backend`` and
            ``run_label`` resolved.
        """
        self.embedder_backend = resolve_embedder_backend(
            self.topology, self.embedder_backend
        )
        self.reranker_backend = resolve_reranker_backend(
            self.topology, self.reranker_backend
        )
        self.run_label = validate_run_label(self.run_label or self._default_label())
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
    def check_openai_reranker(self) -> RagSettings:
        """Fail at start-up when the ``openai`` reranker has no route or model.

        Returns:
            The settings, unchanged.

        Raises:
            ValueError: If ``reranker_backend`` is ``openai`` without
                ``OPENAI_BASE_URL`` or ``RAG_OPENAI_RERANK_MODEL``.
        """
        if self.reranker_backend != "openai":
            return self
        if self.openai_base_url and self.openai_rerank_model:
            return self
        raise ValueError(
            "RAG_RERANKER_BACKEND='openai' posts to {OPENAI_BASE_URL}/rerank: set "
            "OPENAI_BASE_URL to an OpenAI-compatible gateway that has that route "
            "(api.openai.com has none) and RAG_OPENAI_RERANK_MODEL to its model; got "
            f"OPENAI_BASE_URL={self.openai_base_url!r}, "
            f"RAG_OPENAI_RERANK_MODEL={self.openai_rerank_model!r}"
        )

    def workflow_parameters(self) -> WorkflowParameters:
        """Build what ``RagWorkflow`` runs with: retrieval mode, sizes and prompts.

        The query prefix belongs to ``embedder_model``, so only an embedder that
        serves it (``local`` or ``ovms``) gets it.

        Returns:
            The workflow parameters from the retrieval settings and the two
            prompts.

        Example:
            >>> RagWorkflow(deps, RagSettings().workflow_parameters())
        """
        serves_our_model = self.embedder_backend in ("local", "ovms")
        return WorkflowParameters(
            top_k_retrieve=self.top_k_retrieve,
            top_k_rerank=self.top_k_rerank,
            prompts=RagPrompts(system=self.system_prompt, user=self.user_prompt),
            retrieval_mode=self.retrieval_mode,
            rrf_k=self.hybrid_rrf_k,
            query_prefix=self.embedder_query_prefix if serves_our_model else "",
        )

    def rerank_api(self) -> RerankApi:
        """Name the hosted rerank endpoint ``reranker_backend`` stands for.

        Returns:
            Cohere's endpoint, model and key for ``cohere``; for ``openai``, the
            ``/rerank`` route under ``OPENAI_BASE_URL`` with the OpenAI key.

        Raises:
            ValueError: If ``reranker_backend`` is not a hosted rerank API.

        Example:
            >>> RagSettings(reranker_backend="cohere").rerank_api().url
            'https://api.cohere.com/v2/rerank'
        """
        if self.reranker_backend == "cohere":
            return RerankApi(
                self.cohere_rerank_url, self.cohere_rerank_model, self.cohere_api_key
            )
        if self.reranker_backend == "openai":
            url = self.openai_base_url.rstrip("/") + "/rerank"
            return RerankApi(url, self.openai_rerank_model, self.openai_api_key)
        raise ValueError(
            f"RAG_RERANKER_BACKEND={self.reranker_backend!r} is not a hosted rerank "
            "API: expected 'cohere' or 'openai'"
        )

    def _default_label(self) -> str:
        return default_run_label(
            self.topology,
            embedder=self.embedder_backend,
            reranker=self.reranker_backend,
            store=self.vector_store_backend,
            llm=self.llm_backend,
            retrieval=self.retrieval_mode,
        )


# ``RAG_PROMPTS_FILE`` when set, else the ``prompts.yaml`` shipped with us.
def _prompts_path() -> Path:
    override = os.environ.get("RAG_PROMPTS_FILE")
    return Path(override) if override else PACKAGED_PROMPTS


def _prompts_source(
    settings_cls: type[BaseSettings], path: Path
) -> YamlConfigSettingsSource:
    """Build the YAML settings source over the prompts file.

    A missing or unreadable file is an error: pydantic-settings would silently skip
    a missing one, leaving the prompts "Field required" without saying which file
    was meant.

    Args:
        settings_cls: The settings class the source feeds.
        path: The prompts YAML file.

    Returns:
        The YAML source over ``path``.

    Raises:
        FileNotFoundError: If ``path`` is not a file.
        OSError: If ``path`` cannot be read.
    """
    if not path.is_file():
        raise FileNotFoundError(f"prompts file {str(path)!r} does not exist")
    try:
        return YamlConfigSettingsSource(settings_cls, yaml_file=path)
    except OSError as exc:
        raise OSError(f"cannot read prompts file {str(path)!r}: {exc}") from exc
