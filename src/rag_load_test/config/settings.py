"""Runtime configuration from ``RAG_*`` environment variables (or a ``.env`` file).

``RAG_TOPOLOGY`` decides the embedder and reranker backends (the rule lives in
``domain/topology.py``); ``RAG_EMBEDDER_BACKEND`` / ``RAG_RERANKER_BACKEND`` may only
repeat that choice or pick ``fake`` (or ``none`` for the reranker). A
contradiction or a bad ``RAG_RUN_LABEL`` fails construction, so at start-up.

Example:
    >>> settings = RagSettings()  # from the env
    >>> settings = RagSettings(_env_file=None, reranker_backend="fake")  # in tests

Every tunable (retrieval sizes, index, batch sizes, timeouts, ...) is a field here,
so one ``RAG_*`` variable changes it; adapters receive the values from
``adapters/rag_ports.py`` and never read this module. The prompts have no Python
default: they come from ``prompts.yaml`` (or the file ``RAG_PROMPTS_FILE`` names),
the lowest-priority source, so ``RAG_SYSTEM_PROMPT`` / ``RAG_USER_PROMPT`` and
constructor arguments still override them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

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
    RerankerBackend,
    Topology,
    VectorStoreBackend,
    default_run_label,
    resolve_embedder_backend,
    resolve_reranker_backend,
    validate_run_label,
)
from ..domain.workflow_parameters import WorkflowParameters

PACKAGED_PROMPTS = Path(__file__).with_name("prompts.yaml")


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
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    reranker_backend: RerankerBackend = Field(
        default=cast(RerankerBackend, None), validate_default=False
    )
    reranker_model: str = "BAAI/bge-reranker-base"

    ovms_embeddings_url: str = "http://localhost:8002"
    ovms_embeddings_model: str = "bge-small-en-v1.5"
    ovms_rerank_url: str = "http://localhost:8001"
    ovms_rerank_model: str = "bge-reranker-base"
    # "" = no auth, "domino" = the run's token from localhost:8899, else a bearer token
    ovms_auth: str = ""

    llm_backend: LlmBackend = DEFAULT_LLM_BACKEND
    openai_model: str = "gpt-4o-mini"  # OPENAI_API_KEY / OPENAI_BASE_URL via the SDK

    # "fake" = in-memory store seeded with the synthetic corpus when the app starts
    vector_store_backend: VectorStoreBackend = DEFAULT_VECTOR_STORE_BACKEND
    es_url: str = "http://localhost:9200"  # user:password@ in the URL for basic auth
    es_api_key: str = ""
    es_index: str = "rag_passages"
    domino_tracing: bool = False

    # --- workflow: retrieval sizes and prompts (no default: see prompts.yaml) -----
    top_k_retrieve: int = Field(default=20, ge=1)  # candidates from the store
    top_k_rerank: int = Field(default=5, ge=1)  # passages kept and sent to the LLM
    system_prompt: str
    user_prompt: str  # holds {context} and {question}

    # --- in-process models -------------------------------------------------------
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

    def workflow_parameters(self) -> WorkflowParameters:
        """Build the retrieval sizes and prompts ``RagWorkflow`` runs with.

        Returns:
            The workflow parameters from ``top_k_retrieve``, ``top_k_rerank`` and
            the two prompts.

        Example:
            >>> RagWorkflow(deps, RagSettings().workflow_parameters())
        """
        return WorkflowParameters(
            top_k_retrieve=self.top_k_retrieve,
            top_k_rerank=self.top_k_rerank,
            prompts=RagPrompts(system=self.system_prompt, user=self.user_prompt),
        )

    def _default_label(self) -> str:
        return default_run_label(
            self.topology,
            embedder=self.embedder_backend,
            reranker=self.reranker_backend,
            store=self.vector_store_backend,
            llm=self.llm_backend,
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
