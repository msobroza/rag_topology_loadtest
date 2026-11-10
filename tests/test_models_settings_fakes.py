"""Behavioural tests for the dependency-free core: models, settings and fakes."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import pytest
from pydantic import ValidationError

from rag_load_test.adapters.fakes import fake_models
from rag_load_test.adapters.fakes.fake_models import (
    FakeChatModel,
    FakeEmbedder,
    FakeReranker,
)
from rag_load_test.adapters.fakes.in_memory_vector_store import InMemoryVectorStore
from rag_load_test.config.settings import RagSettings
from rag_load_test.domain.errors import RagDependencyError, describe_error
from rag_load_test.domain.passages import Passage, ScoredPassage, top_by_score
from rag_load_test.domain.ports import (
    ChatModelPort,
    EmbedderPort,
    RerankerPort,
    VectorStorePort,
)


def _scored(pid: str, text: str = "", retrieval_score: float = 0.5) -> ScoredPassage:
    return ScoredPassage(id=pid, text=text or pid, retrieval_score=retrieval_score)


# --- models ------------------------------------------------------------------


def test_top_by_score_orders_desc_truncates_and_stamps_float_rerank_score() -> None:
    passages = [_scored("a", retrieval_score=0.1), _scored("b"), _scored("c")]
    kept = top_by_score(passages, scores=[1, 3, 2], top_n=2)  # ints on purpose

    assert [p.id for p in kept] == ["b", "c"]
    assert [p.rerank_score for p in kept] == [3.0, 2.0]
    assert all(type(p.rerank_score) is float for p in kept)
    assert [p.retrieval_score for p in kept] == [0.5, 0.5]
    assert passages[0].rerank_score is None  # inputs are copied, not mutated


def test_rag_dependency_error_message_carries_component_kind_and_target() -> None:
    err = RagDependencyError("reranker", "HTTP 503", target="http://ovms:8001")

    assert err.kind == "unavailable"
    assert str(err) == "reranker unavailable: HTTP 503 (target=http://ovms:8001)"
    assert "timeout" in str(RagDependencyError("llm", "slow", kind="timeout"))


def test_describe_error_names_the_type_and_the_message_when_there_is_one() -> None:
    described = describe_error(TimeoutError("read timed out"))
    assert described == "TimeoutError: read timed out"
    assert describe_error(TimeoutError()) == "TimeoutError"


def test_fakes_subclass_their_ports() -> None:
    assert isinstance(FakeEmbedder(), EmbedderPort)
    assert isinstance(FakeReranker(), RerankerPort)
    assert isinstance(InMemoryVectorStore(), VectorStorePort)
    assert isinstance(FakeChatModel(), ChatModelPort)
    assert not isinstance(FakeChatModel(), VectorStorePort)


def test_a_port_subclass_missing_a_method_cannot_be_instantiated() -> None:
    class HalfReranker(RerankerPort):
        """Implements ``rerank`` but forgets ``ready``."""

        async def rerank(
            self, query: str, passages: Sequence[ScoredPassage], top_n: int
        ) -> list[ScoredPassage]:
            return []

    with pytest.raises(TypeError, match=r"HalfReranker.*ready"):
        HalfReranker()  # type: ignore[abstract]


def test_a_class_with_the_right_methods_but_no_port_base_is_not_a_port() -> None:
    class DuckEmbedder:
        model_name = "duck"

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return []

        async def ready(self) -> tuple[bool, str]:
            return True, "duck"

    assert not isinstance(DuckEmbedder(), EmbedderPort)


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
    # moved or added here on purpose.
    assert sorted(RagSettings.model_fields) == [
        "domino_api_proxy",
        "domino_hardware_tier_id",
        "domino_token_ttl_s",
        "domino_token_url",
        "domino_tracing",
        "embedder_backend",
        "embedder_batch_size",
        "embedder_model",
        "embedder_threads",
        "es_api_key",
        "es_index",
        "es_url",
        "fake_store_seed_documents",
        "http_backoff_s",
        "http_retries",
        "http_timeout_s",
        "llm_backend",
        "model_cpus",
        "model_workers",
        "openai_model",
        "ovms_auth",
        "ovms_embeddings_model",
        "ovms_embeddings_url",
        "ovms_rerank_model",
        "ovms_rerank_url",
        "reranker_backend",
        "reranker_batch_size",
        "reranker_model",
        "reranker_threads",
        "run_label",
        "system_prompt",
        "top_k_rerank",
        "top_k_retrieve",
        "topology",
        "user_prompt",
        "vector_store_backend",
    ]


# --- fakes -------------------------------------------------------------------


async def test_fake_embedder_is_deterministic_unit_norm_and_topical() -> None:
    embedder = FakeEmbedder()
    pie, pie_again, tart, physics = await embedder.embed(
        ["red apple pie", "red apple pie", "red apple tart", "quantum gravity lecture"]
    )

    assert pie == pie_again
    assert math.isclose(math.sqrt(sum(x * x for x in pie)), 1.0, rel_tol=1e-9)
    assert fake_models.cosine(pie, tart) > fake_models.cosine(pie, physics)


async def test_fake_reranker_scores_by_overlap_and_truncates() -> None:
    passages = [
        _scored("off", "gardening tips for spring"),
        _scored("hit", "the elasticsearch cluster shards"),
        _scored("near", "cluster of stars"),
    ]
    kept = await FakeReranker().rerank("elasticsearch cluster", passages, top_n=2)

    assert [p.id for p in kept] == ["hit", "near"]
    assert kept[0].rerank_score is not None
    assert kept[0].rerank_score > 0
    assert kept[0].rerank_score > (kept[1].rerank_score or 0)


async def test_in_memory_store_upserts_idempotently_and_queries_in_order() -> None:
    store = InMemoryVectorStore()
    doc = Passage(id="p1", text="one", metadata={"page": 3, "src": "a.pdf"})
    await store.upsert([doc], [[1.0, 0.0]])
    await store.upsert([doc], [[1.0, 0.0]])  # same id twice -> one row
    await store.upsert(
        [Passage(id="p2", text="two"), Passage(id="p3", text="three")],
        [[0.0, 1.0], [0.6, 0.8]],
    )

    assert await store.count() == 3
    hits = await store.query([1.0, 0.0], top_k=2)
    assert [(h.id, h.retrieval_score) for h in hits] == [("p1", 1.0), ("p3", 0.6)]
    assert hits[0].metadata == {"page": 3, "src": "a.pdf"}


async def test_in_memory_store_is_ready_once_it_holds_a_passage() -> None:
    store = InMemoryVectorStore()
    assert await store.ready() == (False, "0 passages in memory")

    await store.upsert([Passage(id="p1", text="one")], [[1.0, 0.0]])
    assert await store.ready() == (True, "1 passages in memory")


async def test_in_memory_store_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="shorter"):  # zip(strict=True)
        await InMemoryVectorStore().upsert(
            [Passage(id="a", text="a"), Passage(id="b", text="b")], [[1.0]]
        )


async def test_fake_chat_model_echoes_last_user_message_and_counts_calls() -> None:
    llm = FakeChatModel()
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]

    assert await llm.generate(messages) == "[fake-llm] second"
    assert await llm.generate([{"role": "system", "content": "x"}]) == "[fake-llm] "
    assert llm.calls == 2


async def test_fake_chat_model_honours_latency() -> None:
    llm = FakeChatModel(latency_ms=30)
    start = time.perf_counter()
    await llm.generate([{"role": "user", "content": "hi"}])

    assert time.perf_counter() - start >= 0.025
