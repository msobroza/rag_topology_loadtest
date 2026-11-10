"""Runtime configuration from ``RAG_*`` environment variables (or a ``.env`` file).

``RAG_TOPOLOGY`` decides the embedder and reranker backends (the rule lives in
``domain/topology.py``); ``RAG_EMBEDDER_BACKEND`` / ``RAG_RERANKER_BACKEND`` may only
repeat that choice or pick ``fake`` (or ``none`` for the reranker). A
contradiction or a bad ``RAG_RUN_LABEL`` fails construction, so at start-up.

Example::

    settings = RagSettings()                                          # from the env
    settings = RagSettings(_env_file=None, reranker_backend="fake")   # in tests
"""

from __future__ import annotations

from typing import cast

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", extra="ignore", protected_namespaces=()
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
    domino_tracing: bool = False

    @model_validator(mode="after")
    def resolve_topology(self) -> RagSettings:
        """Fill the model backends from the topology and the default run label."""
        self.embedder_backend = resolve_embedder_backend(
            self.topology, self.embedder_backend
        )
        self.reranker_backend = resolve_reranker_backend(
            self.topology, self.reranker_backend
        )
        self.run_label = validate_run_label(self.run_label or self._default_label())
        return self

    def _default_label(self) -> str:
        return default_run_label(
            self.topology,
            embedder=self.embedder_backend,
            reranker=self.reranker_backend,
            store=self.vector_store_backend,
            llm=self.llm_backend,
        )
