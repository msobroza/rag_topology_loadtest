"""``RagSettings``: defaults, the environment, backends from the topology, the pinned field set."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from rag_load_test.config.settings import RagSettings
from rag_load_test.domain.passages import ScoredPassage


def _scored(pid: str, text: str = "", retrieval_score: float = 0.5) -> ScoredPassage:
    return ScoredPassage(id=pid, text=text or pid, retrieval_score=retrieval_score)


# --- settings ----------------------------------------------------------------


def test_settings_defaults(settings_factory: Callable[..., RagSettings]) -> None:
    s = settings_factory()

    assert s.topology == "monolith"
    assert (s.embedder_backend, s.reranker_backend) == ("local", "local")
    assert s.run_label == "monolith"
    assert s.ovms_rerank_url == "http://localhost:8001"
    assert s.ovms_embeddings_url == "http://localhost:8002"
    assert s.es_url == "http://localhost:9200"
    assert s.ovms_auth == ""


def test_settings_read_rag_prefixed_env(
    settings_factory: Callable[..., RagSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_TOPOLOGY", "split-reranker")
    monkeypatch.setenv("RAG_RERANKER_BACKEND", "ovms")
    monkeypatch.setenv("RAG_RUN_LABEL", "split-reranker-fastapi")
    monkeypatch.setenv("RAG_DOMINO_TRACING", "true")

    s = settings_factory()
    assert s.reranker_backend == "ovms"
    assert s.run_label == "split-reranker-fastapi"
    assert s.domino_tracing is True


@pytest.mark.parametrize(
    ("topology", "embedder", "reranker"),
    [
        ("monolith", "local", "local"),
        ("split-reranker", "local", "ovms"),
        ("split-all", "ovms", "ovms"),
    ],
)
def test_settings_resolve_backends_from_the_topology(
    settings_factory: Callable[..., RagSettings],
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    embedder: str,
    reranker: str,
) -> None:
    monkeypatch.setenv("RAG_TOPOLOGY", topology)

    s = settings_factory()
    assert (s.embedder_backend, s.reranker_backend) == (embedder, reranker)
    assert s.run_label == topology


def test_settings_default_run_label_names_the_variant_backends(
    settings_factory: Callable[..., RagSettings],
) -> None:
    s = settings_factory(topology="split-all", reranker_backend="none")

    assert s.run_label == "split-all+reranker=none"


def test_settings_reject_a_backend_that_contradicts_the_topology(
    settings_factory: Callable[..., RagSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_EMBEDDER_BACKEND", "ovms")

    with pytest.raises(ValidationError, match="RAG_EMBEDDER_BACKEND='ovms'") as caught:
        settings_factory()

    assert "set RAG_TOPOLOGY=split-all, or drop RAG_EMBEDDER_BACKEND" in str(
        caught.value
    )


def test_settings_reject_a_bad_run_label(
    settings_factory: Callable[..., RagSettings],
) -> None:
    with pytest.raises(ValidationError, match="RAG_RUN_LABEL='two words'"):
        settings_factory(run_label="two words")


def test_settings_reject_unknown_backend(
    settings_factory: Callable[..., RagSettings],
) -> None:
    with pytest.raises(ValidationError, match="tpu"):
        settings_factory(embedder_backend="tpu")


def test_settings_reject_unknown_topology(
    settings_factory: Callable[..., RagSettings],
) -> None:
    with pytest.raises(ValidationError, match="split-reranker-fastapi"):
        settings_factory(topology="split-reranker-fastapi")


def test_settings_field_set_is_pinned() -> None:
    # Experiments should vary the topology, not grow knobs: adding a field is a
    # decision. The tunables (top-k, index, batch sizes, workers, HTTP, Domino token,
    # seed size, HTTP backoff), the prompts and the thread-budget inputs were
    # moved or added here on purpose; so were the hosted model APIs, Chroma, the
    # retrieval mode, the LLM temperature and the in-process model runtime.
    assert sorted(RagSettings.model_fields) == [
        "bm25_b",
        "bm25_k1",
        "chroma_collection",
        "chroma_path",
        "chroma_url",
        "cohere_api_key",
        "cohere_rerank_model",
        "cohere_rerank_url",
        "domino_api_proxy",
        "domino_hardware_tier_id",
        "domino_token_ttl_s",
        "domino_token_url",
        "domino_tracing",
        "embedder_backend",
        "embedder_batch_size",
        "embedder_model",
        "embedder_query_prefix",
        "embedder_threads",
        "es_api_key",
        "es_index",
        "es_url",
        "fake_store_seed_documents",
        "http_backoff_s",
        "http_retries",
        "http_timeout_s",
        "hybrid_rrf_k",
        "llm_backend",
        "llm_temperature",
        "local_model_runtime",
        "model_cpus",
        "model_workers",
        "openai_api_key",
        "openai_base_url",
        "openai_embedding_model",
        "openai_model",
        "openai_rerank_model",
        "openvino_models_dir",
        "ovms_auth",
        "ovms_embeddings_model",
        "ovms_embeddings_url",
        "ovms_rerank_model",
        "ovms_rerank_url",
        "reranker_backend",
        "reranker_batch_size",
        "reranker_model",
        "reranker_threads",
        "retrieval_mode",
        "run_label",
        "system_prompt",
        "top_k_rerank",
        "top_k_retrieve",
        "topology",
        "user_prompt",
        "vector_store_backend",
    ]
