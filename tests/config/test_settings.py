"""``RagSettings``: the tunables, their validation, and the prompts read from YAML."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_load_test.config.settings import RagSettings
from rag_load_test.domain.prompts import RagPrompts
from rag_load_test.domain.workflow_parameters import WorkflowParameters

SettingsFactory = Callable[..., RagSettings]
SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages. "
    "Cite passages as [n]. If the context is insufficient, say so."
)
USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}"


@pytest.fixture(autouse=True)
def _no_prompt_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RAG_PROMPTS_FILE", "RAG_SYSTEM_PROMPT", "RAG_USER_PROMPT"):
        monkeypatch.delenv(name, raising=False)


# --- tunables ------------------------------------------------------------------


def test_tunable_defaults(settings_factory: SettingsFactory) -> None:
    s = settings_factory()

    assert (s.top_k_retrieve, s.top_k_rerank) == (20, 5)
    assert s.es_index == "rag_passages"
    assert (s.embedder_batch_size, s.reranker_batch_size) == (64, 32)
    assert s.model_workers == 4
    assert (s.http_timeout_s, s.http_retries, s.http_backoff_s) == (30.0, 2, 0.2)
    assert s.domino_token_url == "http://localhost:8899/access-token"
    assert s.domino_token_ttl_s == 240.0
    assert s.fake_store_seed_documents == 200


def test_tunables_read_rag_prefixed_env(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = {
        "RAG_TOP_K_RETRIEVE": "50",
        "RAG_TOP_K_RERANK": "10",
        "RAG_ES_INDEX": "passages_v2",
        "RAG_EMBEDDER_BATCH_SIZE": "16",
        "RAG_RERANKER_BATCH_SIZE": "8",
        "RAG_MODEL_WORKERS": "2",
        "RAG_HTTP_TIMEOUT_S": "5.5",
        "RAG_HTTP_RETRIES": "0",
        "RAG_HTTP_BACKOFF_S": "0.5",
        "RAG_DOMINO_TOKEN_URL": "http://token.test/t",
        "RAG_DOMINO_TOKEN_TTL_S": "60",
        "RAG_FAKE_STORE_SEED_DOCUMENTS": "30",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    s = settings_factory()
    assert (s.top_k_retrieve, s.top_k_rerank, s.es_index) == (50, 10, "passages_v2")
    assert (s.embedder_batch_size, s.reranker_batch_size, s.model_workers) == (16, 8, 2)
    assert (s.http_timeout_s, s.http_retries, s.http_backoff_s) == (5.5, 0, 0.5)
    assert (s.domino_token_url, s.domino_token_ttl_s) == ("http://token.test/t", 60.0)
    assert s.fake_store_seed_documents == 30


def test_top_k_rerank_above_top_k_retrieve_fails_naming_both(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError) as caught:
        settings_factory(top_k_retrieve=3, top_k_rerank=7)

    assert "RAG_TOP_K_RERANK=7" in str(caught.value)
    assert "RAG_TOP_K_RETRIEVE=3" in str(caught.value)


@pytest.mark.parametrize("field", ["top_k_retrieve", "top_k_rerank", "model_workers"])
def test_counts_below_one_are_rejected(
    settings_factory: SettingsFactory, field: str
) -> None:
    with pytest.raises(ValidationError, match=field):
        settings_factory(**{field: 0})


def test_workflow_parameters_carry_the_sizes_and_the_prompts(
    settings_factory: SettingsFactory,
) -> None:
    parameters = settings_factory(
        top_k_retrieve=8, top_k_rerank=2
    ).workflow_parameters()

    assert parameters == WorkflowParameters(
        top_k_retrieve=8,
        top_k_rerank=2,
        prompts=RagPrompts(system=SYSTEM_PROMPT, user=USER_PROMPT),
        retrieval_mode="dense",
        rrf_k=60,
        query_prefix="query: ",
    )


# --- prompts --------------------------------------------------------------------


def test_packaged_prompts_equal_the_historic_strings(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory()

    assert s.system_prompt == SYSTEM_PROMPT
    assert s.user_prompt == USER_PROMPT


def test_env_overrides_one_prompt_and_keeps_the_other(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_SYSTEM_PROMPT", "Answer in French.")

    s = settings_factory()
    assert (s.system_prompt, s.user_prompt) == ("Answer in French.", USER_PROMPT)


def test_init_kwargs_override_the_env_prompt(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_USER_PROMPT", "env {context} {question}")

    s = settings_factory(user_prompt="kwarg {question} {context}")
    assert s.user_prompt == "kwarg {question} {context}"


def test_prompts_file_env_points_at_another_yaml(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text(
        "system_prompt: Be terse.\nuser_prompt: '{question} given {context}'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_PROMPTS_FILE", str(prompts))

    s = settings_factory()
    assert (s.system_prompt, s.user_prompt) == (
        "Be terse.",
        "{question} given {context}",
    )


def test_a_missing_prompts_file_fails_naming_the_path(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "nowhere" / "prompts.yaml"
    monkeypatch.setenv("RAG_PROMPTS_FILE", str(missing))

    with pytest.raises(OSError, match=r"nowhere/prompts\.yaml"):
        settings_factory()


def test_a_user_prompt_without_a_placeholder_fails_at_construction(
    settings_factory: SettingsFactory,
) -> None:
    with pytest.raises(ValidationError, match=r"\{context\}"):
        settings_factory(user_prompt="Question: {question}")


# --- thread budget inputs ---------------------------------------------------------


def test_thread_budget_inputs_default_to_find_out(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOMINO_API_PROXY", raising=False)
    monkeypatch.delenv("DOMINO_HARDWARE_TIER_ID", raising=False)

    s = settings_factory()
    assert (s.model_cpus, s.embedder_threads, s.reranker_threads) == (0, 0, 0)
    assert (s.domino_api_proxy, s.domino_hardware_tier_id) == ("", "")


def test_thread_budget_inputs_read_rag_env_and_domino_variables(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_MODEL_CPUS", "2.5")
    monkeypatch.setenv("RAG_EMBEDDER_THREADS", "3")
    monkeypatch.setenv("RAG_RERANKER_THREADS", "1")
    monkeypatch.setenv("DOMINO_API_PROXY", "http://localhost:8899")
    monkeypatch.setenv("DOMINO_HARDWARE_TIER_ID", "small-k8s")

    s = settings_factory()
    assert (s.model_cpus, s.embedder_threads, s.reranker_threads) == (2.5, 3, 1)
    assert s.domino_api_proxy == "http://localhost:8899"
    assert s.domino_hardware_tier_id == "small-k8s"


def test_domino_variables_are_not_read_with_the_rag_prefix(
    settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOMINO_API_PROXY", raising=False)
    monkeypatch.setenv("RAG_DOMINO_API_PROXY", "http://wrong")

    assert settings_factory().domino_api_proxy == ""


def test_domino_variables_can_be_passed_by_field_name(
    settings_factory: SettingsFactory,
) -> None:
    s = settings_factory(domino_api_proxy="http://p", domino_hardware_tier_id="t")
    assert (s.domino_api_proxy, s.domino_hardware_tier_id) == ("http://p", "t")


@pytest.mark.parametrize(
    "field", ["model_cpus", "embedder_threads", "reranker_threads"]
)
def test_negative_thread_budget_inputs_are_rejected(
    settings_factory: SettingsFactory, field: str
) -> None:
    with pytest.raises(ValidationError, match=field):
        settings_factory(**{field: -1})
