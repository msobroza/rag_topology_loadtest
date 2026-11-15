"""``RagSettings``: defaults, the environment, backends from the topology, the pinned field set."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from rag_load_test.config.rag_settings import RagSettings

EMBED_URL = "http://embedder/v3/embeddings"
RERANK_URL = "http://reranker/v3/rerank"


def backends(settings: RagSettings) -> tuple[str, str]:
    components = settings.components
    return components["embedder"].backend, components["reranker"].backend


# --- settings ----------------------------------------------------------------


def test_settings_defaults(settings_factory: Callable[..., RagSettings]) -> None:
    s = settings_factory()

    assert s.topology == "monolith"
    assert backends(s) == ("local", "local")
    assert s.run_label == "monolith"
    assert s.es_url == "http://localhost:9200"
    keys = [component.api.api_key for component in s.components.values()]
    assert keys == ["", "", ""]  # never in the file


def test_settings_read_rag_prefixed_env(
    settings_factory: Callable[..., RagSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_TOPOLOGY", "split-reranker")
    monkeypatch.setenv("RAG_COMPONENTS__RERANKER__SERVED__URL", RERANK_URL)
    monkeypatch.setenv("RAG_RUN_LABEL", "split-reranker-fastapi")
    monkeypatch.setenv("RAG_DOMINO_TRACING", "true")

    s = settings_factory()
    assert s.components["reranker"].backend == "served"
    assert s.components["reranker"].served.url == RERANK_URL
    assert s.run_label == "split-reranker-fastapi"
    assert s.domino_tracing is True


@pytest.mark.parametrize(
    ("topology", "embedding", "rerank"),
    [
        ("monolith", "local", "local"),
        ("split-reranker", "local", "served"),
        ("split-all", "served", "served"),
    ],
)
def test_the_topology_moves_a_local_component_behind_a_model_server(
    settings_factory: Callable[..., RagSettings],
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
    embedding: str,
    rerank: str,
) -> None:
    # The workflow file says local for both; the topology variable wins over it.
    monkeypatch.setenv("RAG_TOPOLOGY", topology)
    monkeypatch.setenv("RAG_COMPONENTS__EMBEDDER__SERVED__URL", EMBED_URL)
    monkeypatch.setenv("RAG_COMPONENTS__RERANKER__SERVED__URL", RERANK_URL)

    s = settings_factory()
    assert backends(s) == (embedding, rerank)
    assert s.run_label == topology


def test_the_topology_places_only_the_embedder_and_the_reranker(
    settings_factory: Callable[..., RagSettings],
) -> None:
    other = {"type": "reranker", "backend": "local", "source_id": "org/other"}

    s = settings_factory(
        topology="split-all",
        components__embedder__served__url=EMBED_URL,
        components__reranker__served__url=RERANK_URL,
        components={"other": other},
    )

    assert s.components["other"].backend == "local"


def test_a_backend_off_local_is_kept_whatever_the_topology(
    settings_factory: Callable[..., RagSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_TOPOLOGY", "split-all")
    monkeypatch.setenv("RAG_COMPONENTS__EMBEDDER__BACKEND", "fake")
    monkeypatch.setenv("RAG_COMPONENTS__RERANKER__BACKEND", "none")

    assert backends(settings_factory()) == ("fake", "none")


def test_an_endpoint_we_do_not_run_is_kept_whatever_the_topology(
    settings_factory: Callable[..., RagSettings],
) -> None:
    # A topology serves the components of ours; an api endpoint stays where it is.
    s = settings_factory(
        topology="split-all",
        components__embedder__backend="api",
        components__embedder__api__url=EMBED_URL,
        components__reranker__served__url=RERANK_URL,
    )

    assert backends(s) == ("api", "served")
    assert s.run_label == "split-all+embedding=api"


def test_settings_default_run_label_names_the_variant_backends(
    settings_factory: Callable[..., RagSettings],
) -> None:
    s = settings_factory(
        topology="split-all",
        components__embedder__served__url=EMBED_URL,
        components__reranker__backend="none",
        components__llm__backend="fake",
    )

    assert s.run_label == "split-all+rerank=none+generation=fake"


def test_settings_reject_a_bad_run_label(
    settings_factory: Callable[..., RagSettings],
) -> None:
    with pytest.raises(ValidationError, match="RAG_RUN_LABEL='two words'"):
        settings_factory(run_label="two words")


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
    # in-process model runtime, and the two limits of calibration. The model
    # components (the embedder, the reranker, the LLM) are the workflow file's
    # components block, each with its blocks of settings; the retrieval mode,
    # the sizes and the prompts are settings of the stages of its graph.
    assert sorted(RagSettings.model_fields) == [
        "calibration_budget_s",
        "calibration_rounds",
        "chroma_collection",
        "chroma_path",
        "chroma_url",
        "components",
        "domino_api_proxy",
        "domino_hardware_tier_id",
        "domino_token_ttl_s",
        "domino_token_url",
        "domino_tracing",
        "es_api_key",
        "es_index",
        "es_text_analyzer",
        "es_url",
        "fake_store_seed_documents",
        "graph",
        "http_backoff_s",
        "http_retries",
        "http_timeout_s",
        "local_model_runtime",
        "model_cpus",
        "model_workers",
        "openvino_models_dir",
        "run_label",
        "topology",
        "vector_store_backend",
    ]


def test_no_flat_variable_of_a_model_component_is_read(
    settings_factory: Callable[..., RagSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "RAG_EMBEDDING_BACKEND": "fake",
        "RAG_RERANK_BACKEND": "none",
        "RAG_GENERATION_BACKEND": "fake",
        "RAG_RERANK_MODEL": "org/other",
        "RAG_EMBEDDING_API_MODEL": "other",
        "RAG_GENERATION_TIMEOUT_S": "1",
    }.items():
        monkeypatch.setenv(name, value)

    s = settings_factory()

    assert [c.backend for c in s.components.values()] == ["local", "local", "api"]
    reranker = s.components["reranker"]
    assert reranker.source_id == "antoinelouis/crossencoder-camembert-base-mmarcoFR"
    assert s.components["embedder"].alias == "snowflake-arctic-embed-l-v2.0"
    assert s.components["llm"].http.timeout_s == 120.0
