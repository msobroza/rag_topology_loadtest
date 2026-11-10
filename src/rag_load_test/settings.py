"""Runtime configuration from ``RAG_*`` environment variables (or a ``.env`` file).

The topology is chosen purely by ``RAG_EMBEDDER_BACKEND`` / ``RAG_RERANKER_BACKEND``:
``local`` runs the model in-process, ``ovms`` calls an OpenVINO Model Server app.

Example::

    settings = RagSettings()                                          # from the env
    settings = RagSettings(_env_file=None, reranker_backend="fake")   # in tests
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    topology: str = "monolith"  # label echoed in responses so Locust can tag runs

    embedder_backend: Literal["local", "ovms", "fake"] = "local"
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    reranker_backend: Literal["local", "ovms", "fake", "none"] = "local"
    reranker_model: str = "BAAI/bge-reranker-base"

    ovms_embeddings_url: str = "http://localhost:8002"
    ovms_embeddings_model: str = "bge-small-en-v1.5"
    ovms_rerank_url: str = "http://localhost:8001"
    ovms_rerank_model: str = "bge-reranker-base"
    # "" = no auth, "domino" = the run's token from localhost:8899, else a bearer token
    ovms_auth: str = ""

    llm_backend: Literal["openai", "fake"] = "openai"
    openai_model: str = "gpt-4o-mini"  # OPENAI_API_KEY / OPENAI_BASE_URL via the SDK

    es_url: str = "http://localhost:9200"  # user:password@ in the URL for basic auth
    es_api_key: str = ""
    domino_tracing: bool = False
