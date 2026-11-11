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
    assert (s.embedding_backend, s.rerank_backend) == ("local", "local")
    assert s.run_label == "monolith"
    assert (s.rerank_api_url, s.embedding_api_url) == ("", "")
    assert s.es_url == "http://localhost:9200"
    assert (s.rerank_api_key, s.embedding_api_key, s.generation_api_key) == ("",) * 3


def test_settings_read_rag_prefixed_env(
    settings_factory: Callable[..., RagSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_TOPOLOGY", "split-reranker")
    monkeypatch.setenv("RAG_RERANK_BACKEND", "api")
    monkeypatch.setenv("RAG_RERANK_API_URL", "http://reranker:8011/v3/rerank")
    monkeypatch.setenv("RAG_RUN_LABEL", "split-reranker-fastapi")
    monkeypatch.setenv("RAG_DOMINO_TRACING", "true")

    s = settings_factory()
    assert s.rerank_backend == "api"
    assert s.run_label == "split-reranker-fastapi"
    assert s.domino_tracing is True


@pytest.mark.parametrize(
    ("topology", "embedding", "rerank"),
    [
        ("monolith", "local", "local"),
        ("split-reranker", "local", "api"),
        ("split-all", "api", "api"),
    ],
)
def test_settings_resolve_backends_from_the_topology(
    settings_factory: Callable[..., RagSettings],
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    embedding: str,
    rerank: str,
) -> None:
    monkeypatch.setenv("RAG_TOPOLOGY", topology)
    monkeypatch.setenv("RAG_EMBEDDING_API_URL", "http://embedder/v3/embeddings")
    monkeypatch.setenv("RAG_RERANK_API_URL", "http://reranker/v3/rerank")

    s = settings_factory()
    assert (s.embedding_backend, s.rerank_backend) == (embedding, rerank)
    assert s.run_label == topology


def test_settings_default_run_label_names_the_variant_backends(
    settings_factory: Callable[..., RagSettings],
) -> None:
    s = settings_factory(
        topology="split-all",
        embedding_api_url="http://embedder/v3/embeddings",
        rerank_backend="none",
        generation_backend="fake",
    )

    assert s.run_label == "split-all+rerank=none+generation=fake"


def test_settings_reject_a_backend_that_contradicts_the_topology(
    settings_factory: Callable[..., RagSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_TOPOLOGY", "split-all")
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "local")

    with pytest.raises(
        ValidationError, match="RAG_EMBEDDING_BACKEND='local'"
    ) as caught:
        settings_factory()

    assert (
        "set RAG_TOPOLOGY=monolith or RAG_TOPOLOGY=split-reranker, "
        "or drop RAG_EMBEDDING_BACKEND"
    ) in str(caught.value)


def test_settings_reject_a_bad_run_label(
    settings_factory: Callable[..., RagSettings],
) -> None:
    with pytest.raises(ValidationError, match="RAG_RUN_LABEL='two words'"):
        settings_factory(run_label="two words")


def test_settings_reject_unknown_backend(
    settings_factory: Callable[..., RagSettings],
) -> None:
    with pytest.raises(ValidationError, match="tpu"):
        settings_factory(embedding_backend="tpu")


def test_settings_reject_unknown_topology(
    settings_factory: Callable[..., RagSettings],
) -> None:
    with pytest.raises(ValidationError, match="split-reranker-fastapi"):
        settings_factory(topology="split-reranker-fastapi")


def test_settings_field_set_is_pinned() -> None:
    # Experiments should vary the topology, not grow knobs: adding a field is a
    # decision. The tunables (top-k, index, batch sizes, workers, HTTP, Domino token,
    # seed size, HTTP backoff), the prompts and the thread-budget inputs were
    # moved or added here on purpose; so were Chroma, the retrieval mode and the
    # in-process model runtime. Each model component (embedding, rerank,
    # generation) has one block of settings, its endpoint in four *_api_* fields.
    assert sorted(RagSettings.model_fields) == [
        "chroma_collection",
        "chroma_path",
        "chroma_url",
        "domino_api_proxy",
        "domino_hardware_tier_id",
        "domino_token_ttl_s",
        "domino_token_url",
        "domino_tracing",
        "embedding_api_format",
        "embedding_api_key",
        "embedding_api_model",
        "embedding_api_url",
        "embedding_backend",
        "embedding_batch_size",
        "embedding_model",
        "embedding_query_prefix",
        "embedding_threads",
        "es_api_key",
        "es_index",
        "es_text_analyzer",
        "es_url",
        "fake_store_seed_documents",
        "generation_api_format",
        "generation_api_key",
        "generation_api_model",
        "generation_api_url",
        "generation_backend",
        "generation_temperature",
        "generation_timeout_s",
        "http_backoff_s",
        "http_retries",
        "http_timeout_s",
        "hybrid_rrf_k",
        "local_model_runtime",
        "model_cpus",
        "model_workers",
        "openvino_models_dir",
        "rerank_api_format",
        "rerank_api_key",
        "rerank_api_model",
        "rerank_api_url",
        "rerank_backend",
        "rerank_batch_size",
        "rerank_model",
        "rerank_threads",
        "retrieval_mode",
        "run_label",
        "system_prompt",
        "top_k_rerank",
        "top_k_retrieve",
        "topology",
        "user_prompt",
        "vector_store_backend",
    ]
